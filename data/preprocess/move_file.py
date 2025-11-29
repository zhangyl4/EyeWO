import paramiko
import os

def get_file_list_from_txt(txt_file_path):
    """从 txt 文件中读取要搬运的文件路径列表"""
    with open(txt_file_path, 'r') as file:
        files = [line.strip().split('.')[0] + '.mp4' for line in file.readlines()]
    return files

# def get_file_list_from_txt(txt_file_path):
#     """从 txt 文件中读取要搬运的文件路径列表"""
#     with open(txt_file_path, 'r') as file:
#         files = [line.strip() for line in file.readlines()]
#     return files

def ssh_connect(host, port, username, password):
    """连接到远程服务器"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=username, password=password)
    return client

def ssh_connect_identity_file(host, port, username, identity_file_path):
    """使用密钥文件连接到远程服务器"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=username, key_filename=identity_file_path)
    return client
    
    
def scp_file_to_remote(ssh_client, local_path, remote_path):
    """使用 SCP 将文件从本地传输到远程"""
    ftp_client = ssh_client.open_sftp()
    try:
        print(f"正在搬运文件: {local_path} 到 {remote_path}")
        ftp_client.put(local_path, remote_path)
        print("搬运完成")
    except Exception as e:
        print(f"搬运文件失败: {e}")
    finally:
        ftp_client.close()

def main(cluster_a_info, cluster_b_info, folder_path, txt_file_path, target_folder):
    # 从 txt 文件中获取要搬运的文件列表
    files_to_transfer = get_file_list_from_txt(txt_file_path)

    # 连接到集群 A
    # print("连接到集群 A")
    # client_a = ssh_connect_identity_file(cluster_a_info['host'], cluster_a_info['port'], 
    #                        cluster_a_info['username'], cluster_a_info['identity_file'])

    # 连接到集群 B
    print("连接到集群 B")
    if 'password' in cluster_b_info:
        client_b = ssh_connect(cluster_b_info['host'], cluster_b_info['port'], 
                           cluster_b_info['username'], cluster_b_info['password'])
    else:
        client_b = ssh_connect_identity_file(cluster_b_info['host'], cluster_b_info['port'], 
                           cluster_b_info['username'], cluster_b_info['identity_file'])
    # create target folder
    stdin, stdout, stderr = client_b.exec_command(f'mkdir -p {target_folder}')
    print('创建目标文件夹')
    
    # 遍历文件并搬运
    for file_name in files_to_transfer:
        local_file_path = os.path.join(folder_path, file_name)
        
        # 在集群 B 的目标文件夹创建文件路径
        remote_file_path = os.path.join(target_folder, file_name)
        
        # 搬运文件到集群 B
        scp_file_to_remote(client_b, local_file_path, remote_file_path)

    # 关闭 SSH 连接
    # client_a.close()
    client_b.close()
    print("全部文件搬运完成")

if __name__ == "__main__":
    # 集群 A 和集群 B 的 SSH 连接信息
    cluster_a_info = {
        'host': '10.15.89.211',
        'port': 6286,  # 默认 SSH 端口
        'username': 'zhangyl',
        'identity_file': r"C:\Users\张宇麟\.ssh\id_ed25519"  # 身份文件路径
    }

    # ai cluster
    # cluster_b_info = {
    #     'host': '10.15.89.192',
    #     'port': 22112,  # 默认 SSH 端口
    #     'username': 'zhangyl4',
    #     'password': 'sist'
    # }
    
    # a40
    cluster_b_info = {
        'host': '10.15.88.45',
        'port': 20905,  # 默认 SSH 端口
        'username': 'root',
        'identity_file': "/2022233235/videollm-online/data/preprocess/id_ed25519"  # 身份文件路径
    }

    # 文件夹路径和 txt 文件路径
    folder_path = '/2022233235/datasets/ego4d/full_scale_2fps'
    txt_file_path = '/2022233235/videollm-online/data/preprocess/file_list.txt'
    target_folder = '/root/videollm-online/datasets/ego4d/v2/full_scale_2fps'

    # 开始搬运文件
    main(cluster_a_info, cluster_b_info, folder_path, txt_file_path, target_folder)
