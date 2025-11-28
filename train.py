from dataclasses import asdict

from models import build_model_and_tokenizer, parse_args
from data import build_concat_train_dataset, build_eval_dataset_dict, get_data_collator, get_compute_metrics_dict
from engine import TrainerWithGenToEval



import torch, os, transformers, logging
import torch.distributed as dist
def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, **kwargs):
    """Collects the state dict and dump to disk."""

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {kwargs.get('only_modules_to_ft', None)}")
    if len(kwargs.get('only_modules_to_ft', None)) > 0:
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), kwargs.get('only_modules_to_ft', None))
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa



def train():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(is_training=True, **asdict(args))
    train_dataset = build_concat_train_dataset(tokenizer=tokenizer, **asdict(args))
    eval_dataset_dict = build_eval_dataset_dict(tokenizer=tokenizer, **asdict(args))
    data_collator = get_data_collator(tokenizer=tokenizer, **asdict(args))
    compute_metrics_dict = get_compute_metrics_dict(dataset_dict=eval_dataset_dict, tokenizer=tokenizer, **asdict(args))

    args.gradient_checkpointing_kwargs = {'use_reentrant': False}
    trainer = TrainerWithGenToEval(
        model=model, tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_dict,
        data_collator=data_collator,
        compute_metrics=compute_metrics_dict,
    )
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    safe_save_model_for_hf_trainer(trainer, **asdict(args))

    if eval_dataset_dict is not None:
        metrics = {}
        for eval_dataset_name, eval_dataset in eval_dataset_dict.items():
            trainer.compute_metrics = compute_metrics_dict[eval_dataset_name]
            metrics.update(
                trainer.evaluate(
                    eval_dataset=eval_dataset,
                    metric_key_prefix=f"eval_{eval_dataset_name}",
                )
            )
        print(metrics)

if __name__ == "__main__":
    train()
