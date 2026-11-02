Meta-AMS 完整论文资料：Git 交接入口
整理日期：2026-11-03

仓库：https://github.com/hs997/densact
资料分支：agent/ams-paper-handoff-20261103
当前资料目录：AMS/

首次在另一台电脑获取
  git clone --single-branch --branch agent/ams-paper-handoff-20261103 https://github.com/hs997/densact.git AMS-workspace
  cd AMS-workspace

以后更新（在这个新克隆目录内执行）
  git pull --ff-only

如果已有其他用途的 densact 工作目录，推荐另行克隆到 AMS-workspace，避免切换分支影响未提交工作。
本分支保留既有 AMS Git 历史，并新增完整论文交接目录；未把服务器主工作区全部未提交改动混入提交。
仓库顶层 source/、scripts/ 是已有分支基线；最新完成队列的源码证据在下述哈希验证快照中。

给另一台电脑的 GPT/Codex 的读取顺序
  1. 本文件 AMS/GIT_README_zh.txt。
  2. AMS/results_20261102/PAPER_INTEGRATION_zh.txt 与 README_zh.txt。
  3. AMS/results_20261102/ams_results_20261102.pdf 和对应 .tex。
  4. AMS/results_20261102/data/analysis_results.json、selection_outcomes.csv、full_ladder_summary.csv。
  5. AMS/results_20261102/evidence/source_snapshot/ 与 data/queue_manifest.json：当前完成队列的源码及配置证据。
  6. AMS/delivery_20261103/ams_debug_notes_git_20261103.md：本轮Git整理时的最新工作记录。
  7. 原稿 AMS/ams_paper.pdf、ams_paper.tex、参考文献、CODE_LOGIC.md 和其他历史交接资料。

原论文 PDF 和同名 TeX 不是完全对应的版本。新增实验补充稿没有代替正式理论审查或自动重写原论文。
旧 README.md、旧 project_notes 和旧源码快照保留为历史资料，不能将其过时状态覆盖新实验结果。
新六算法数据、旧单训练链、旧消融及修复续训属于不同队列，必须分别解释，不能拼接统计。

本分支包含
  原桌面 AMS 全部资料 + 2026-11-02 实验补充 + 最新工作记录 + 文件哈希清单。
  原来两处绝对路径目录链接已展开为真实文件；解压/克隆后不依赖服务器软链接。
  新补充包含450份最终评估汇总、104块训练/门禁记录、39份源码快照、表图和代表性视频。
  没有额外纳入全量Isaac Sim、USD资产、所有训练checkpoint及全部逐步评估大文件。
  因此这是论文撰写和复核交接仓库，不是安装后立即可运行仿真训练的完整软件发行版。

可选：校验新补充文件（只需Python标准库）
  python AMS/results_20261102/scripts/verify_package.py

可选：复算表图（建议先复制results_20261102到工作副本）
  python -m pip install -r AMS/results_20261102/requirements.txt
  python AMS/results_20261102/scripts/build_ackermann_paper_results.py --output AMS/results_20261102 --rebuild
  重算会更新派生表图及时间戳；如需保持原始交付哈希，请在副本运行。

邮件请求已由用户取消，没有发送邮件。本轮交付改为Git资料分支。
