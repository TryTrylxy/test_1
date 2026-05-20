import subprocess
import os
import time
import json

# ===================== 配置你的环境 =====================
PYTHONPATH = "/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_07"
os.chdir(PYTHONPATH)

# 你要按顺序执行的命令
TASKS = [
    {
        "config": "./configs/vitb32/vg/cusa_restate_sl1.yaml",
        "log": "/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_07/train.log",
    },
    {
        "config": "./configs/vitb32/vg/cusa_restate_sl2.yaml",
        "log": "/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_07/train.log",
    },
    {
        "config": "./configs/vitb32/vg/cusa_restate_sl3.yaml",
        "log": "/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_07/train.log",
    },
]

# ===================== 按顺序执行 =====================
env = os.environ.copy()
env["PYTHONPATH"] = PYTHONPATH
config_fig = [
    {
        "do_img_neg": False,
        "do_neg": True,
        "do_TxT": True,
        "do_abc": True,
    },
]

for idx, task in enumerate(TASKS, 1):
    print(idx)
    if idx != 2:
        continue
    for idy, fig_dict in enumerate(config_fig, 1):
        config = task["config"]
        log_file = task["log"]
        fig_str = json.dumps(fig_dict)

        # 拼接自动覆盖参数

        # 最终命令（自动覆盖 config 里的值）
        cmd = f'python retrieval.py --config {config} --fig \'{fig_str}\' > {log_file} 2>&1'

        print(f"\n======================================")
        print(f"🚀 运行 Config {idx} | 组合 {idy}")
        print(f"📌 参数：{fig_dict}")
        print(cmd)

        result = subprocess.run(cmd, env=env, shell=True, executable="/bin/bash")

        if result.returncode == 0:
            print(f"✅ 完成")
        else:
            print(f"❌ 失败")
            
        time.sleep(10)

print("\n🎉 所有任务执行完毕！")