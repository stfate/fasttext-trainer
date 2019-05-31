import os
import subprocess
import sys
import tempfile


__neologd_repo_name = "mecab-ipadic-neologd"
__neologd_repo_url = "https://github.com/neologd/mecab-ipadic-neologd.git"


def download_neologd(dic_path):
    dic_path = os.path.abspath(dic_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = f"git clone --depth 1 {__neologd_repo_url}"
        subprocess.run(cmd.split(), stdout=sys.stdout, cwd=temp_dir)
        neologd_dir_path = os.path.join(temp_dir, __neologd_repo_name)
        cmd = f"./bin/install-mecab-ipadic-neologd -y -u -p {dic_path}"
        subprocess.run(cmd.split(), stdout=sys.stdout, cwd=neologd_dir_path)
