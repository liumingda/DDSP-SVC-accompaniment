import os
import subprocess
from tqdm import tqdm
import random

def run_vc_batch(source_txt, target_txt, output_base="./logs/svc_result", model_path="./exp/reflow-test/model_100000.pt"):
    with open(source_txt, "r", encoding="utf-8") as f:
        source_wavs = [line.strip() for line in f if line.strip()]

    with open(target_txt, "r", encoding="utf-8") as f:
        target_wavs = [line.strip() for line in f if line.strip()]
    cwd = os.getcwd()

    for op_source_wav in tqdm(source_wavs, ncols=100):
        source_wav = cwd + "/svc_result/source_data/" + op_source_wav

        # 随机选择100个（不足100则全选）
        sampled_target_wavs = random.sample(target_wavs, min(100, len(target_wavs)))

        for sam_target_wav in sampled_target_wavs:
            target_wav = cwd + "/svc_result/target_data/" + sam_target_wav
            # print(cwd)
            wav_name = target_wav.split("/")[-1]
            # print(target_wav)

        # for target_wav in target_wavs:
        #     # print(target_wav.split("/"))
        #     # singer_id = target_wav.split("/")[-2]
        #     print(cwd)
        #     wav_name = target_wav.split("/")[-1]

        #     print(target_wav)
        #     exit()
            # target_rel = os.path.relpath(target_wav, start=os.path.commonpath([target_wav, output_base]))
            target_rel_prefix = os.path.splitext(wav_name)[0]  # 去掉扩展名
        # print(target_rel_prefix)
        # exit()
        
        # for source_wav in source_wavs:
            source_base = os.path.splitext(os.path.basename(source_wav))[0]  # 去掉路径和扩展名
            # print(source_base)
            # exit()
            out_name = f"{target_rel_prefix}_svc_{source_base}.wav"
            out_path = os.path.join(output_base, target_rel_prefix, out_name)


            # 确保输出目录存在
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            cmd = [
                "python", "main_reflow_copy_test.py",
                "-i", source_wav,
                "-m", model_path,
                "-o", out_path,
                "-k", "0",
                "-tw", target_wav
            ]
            print(f"👉 正在处理: {out_path}")
            subprocess.run(cmd)

if __name__ == "__main__":
    cwd = os.getcwd()
    run_vc_batch(
        source_txt= cwd + "/" + "svc_result/source_wavs.txt",
        target_txt=cwd + "/" + "svc_result/target_data.txt",
        output_base="./svc_result/result"
    )