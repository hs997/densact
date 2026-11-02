Meta-AMS 已完成实验：论文结果补充包
日期：2026-11-02

入口
  ams_results_20261102.pdf：按原 AMS 的 IEEEtran 双栏格式编排的英文实验补充稿。
  ams_results_20261102.tex：可编辑源码；tables/、figures/ 和 references/ 分目录保存。
  PAPER_INTEGRATION_zh.txt：结果解释、论文衔接位置和后续补证据事项。

范围
  数据来自 ackermann_completion_20261027/uniform_six 的已完成队列。
  6 种方法 × 3 个共同 BC 起点下的适应训练种子 × clean 和 8 档攻击。
  104 个训练块，4160 次更新，450 份最终数值评估，162 套原始展示证据。
  本轮没有运行新的训练或仿真，也没有修改算法、门禁、原始结果或原论文。
  本包为实验结果补充，不是完成理论重写的正式整篇论文。

关键结论
  Meta-AMS hard-b 完成率：93.06%，训练种子均值样本标准差 1.20 个百分点；3/3 通过。
  clean 到 hard-a 全部 100%；全部最终评估的记录安全违规率为零。
  三个适应训练种子共用一次 BC 拟合；最终评估条件也用于候选选模。
  早停、回退及不同实际预算详见 selection_outcomes.csv；不要遗漏失败分支。
  五项事后配对精确检验均 p=0.25，Holm 校正均为 1.00；不能写统计显著最优。

无需 Isaac Sim 的结果复算
  进入本目录，在独立 Python 环境安装 requirements.txt 中依赖后运行：
    python scripts/test_ackermann_paper_results.py
    python scripts/build_ackermann_paper_results.py --output . --rebuild
  该命令读取已导出的 CSV 和代表性轨迹，重新生成派生统计、表格、图和派生产物哈希。
  不调用训练、评估或 GPU，不依赖服务器 checkpoint 路径。
  rebuild 会覆盖本补充目录内的派生表图；若需保持交付包原始校验值，请在副本中复算。
  科学绘图 PDF 的创建时间可能变化，复算后的文件哈希不必逐字节等于交付哈希。
  精确检验和 Holm 校正由脚本自行实现；没有运行 SciPy/statsmodels。
  它们的官方方法文档仅作为统计方法参考，见 references/statistical_methods.bib。

编译 PDF
  本次使用 Tectonic 0.17.0（IEEEtran，XeTeX + BibTeX 自动循环）。
  已安装 tectonic 时运行：
    tectonic --keep-logs ams_results_20261102.tex
  当前服务器的实际命令：
    /home/cnc/SSD_1T/xzw/ams_runtime/paper_tools/tectonic_0.17.0/tectonic --keep-logs ams_results_20261102.tex
  初次编译可能需要联网下载 TeX 宏包；编译器未装入训练环境，也未打入本 zip。
  runtime_versions.json 保存分析依赖版本。compile.log 保存成功编译输出。

数据与证据导航
  data/evaluations.csv：450 个最终评估记录及来源路径、模型身份、门禁结果。
  data/full_ladder_summary.csv：54 个方法—强度组合及三个训练种子分项。
  data/selection_outcomes.csv：18 个流程的训练预算、保留模型和 BC 回退情况。
  data/training_stages.csv、training_rewards.csv：104 个块及 4160 次更新记录。
  data/hard_b_paired_comparisons.csv、sign_flip_distribution.csv：效应量和全部 40 个置换统计量。
  data/input_manifest.json、queue_manifest.json：原始路径、配置和来源哈希。
  evidence/evaluations/：450 份原始汇总 CSV 与 verified.json。
  evidence/training/：104 块训练和门禁记录。
  evidence/source_snapshot/：已按队列清单哈希核验的历史源码快照，39 个文件。
  evidence/representative_s0_hard_b/：固定选取 s0 hard-b 的原始视频、真实轨迹、俯视图。
  package_manifest.json：本包所有交付文件的 SHA256（不包含它自身）。
  解压后可运行 python scripts/verify_package.py 校验交付文件的完整性。

原始大文件的边界
  本包包含重算这些表图所需的数据和一组原始视频，但不包含全部逐步评估 CSV、
  全部 162 个视频或训练 checkpoint；不能称为完整实验原始归档。
  完整服务器数据位于：
    /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027
  checkpoint 具体路径在 data/selection_outcomes.csv 中。
  原论文 PDF 和 TeX 内容不完全对应；整篇论文合并前须定位相应版本的源文件。
  原件身份见 original_paper_identity.json。本次新包不覆盖旧 AMS.zip。
