# 两任务四角色试验及病例准入诊断（2026-10-04）

CartPole/Acrobot、四角色、seed8/9/10，固定2,048训练环境步、三次evaluation与两次独立MC行为rollout。病例容量128，top-k8；无早停/阈值调参。Acrobot真实环境6维观测、3动作、500步上限，奖励为-1直到成功终止；物理回报上界0与episode步数上限分开，未设未经确认的竞争性门槛。

冻结源码39906b4043b44b338e52b2622c40ed6689c11313c8874f16f76d5d998a1f1650，ZIP SHA256 b88c3d010ea81d62aaad549aeebcdf5d376b879726d62fa38d5a5054024fed69。24运行/48状态安全载入，96 MC查询/768实际fixed-set删除干预复算；恢复actor病例数量/动作覆盖/评价就绪状态，恢复critic实际活动IDs、所有weights/feature-distance terms/预测/损失/C/H/Q。160测试通过。

| 任务 | actor/critic | 回报均值 [seed范围] | MC MSE均值 | NN actor就绪seed | actor保留数 |
|---|---|---:|---:|---:|---|
| acrobot | mlp/mlp | -500.000 [-500.000,-500.000] | 7001.642 | 3/3 | [None, None, None] |
| acrobot | mlp/nnknn | -500.000 [-500.000,-500.000] | 4188.124 | 3/3 | [None, None, None] |
| acrobot | nnknn/mlp | -458.000 [-458.000,-458.000] | 6930.977 | 0/3 | [0, 0, 0] |
| acrobot | nnknn/nnknn | -458.000 [-458.000,-458.000] | 4117.321 | 0/3 | [0, 3, 0] |
| cartpole | mlp/mlp | 173.444 [72.667,349.333] | 279.743 | 3/3 | [None, None, None] |
| cartpole | mlp/nnknn | 80.333 [64.667,102.333] | 47.097 | 3/3 | [None, None, None] |
| cartpole | nnknn/mlp | 120.444 [89.667,139.000] | 1848.107 | 3/3 | [128, 128, 128] |
| cartpole | nnknn/nnknn | 145.111 [86.000,246.000] | 1013.391 | 3/3 | [128, 128, 128] |

Acrobot六个NN actor运行均未就绪，其中五个没有病例，另一个仅3病例、动作覆盖[2,1,0]。NN actor的-458评价均值来自未就绪的均匀action-space采样回退，不能算作学到的病例策略优于MLP。真实loss日志中正原始优势准入与2,048训练样本核对完成；不是根据负回报推测空库。

CartPole NN/NN均值145.111，较512步pilot均值202.556更低；同seed小预算变化也是探索结果，增加训练没有稳定修复。本批MLP/MLP173.444，NN/MLP120.444，MLP/NN80.333。不同策略访问分布使MC误差不能直接作为同数据排名，环境步一致不等于计算成本一致。

保留AGENTS要求的原始正优势准入，未偷偷改为归一化优势或添加随机经验。下一步检查暖启动baseline与GAE标定、每动作准入覆盖以及训练MC/GAE贡献流，设计明确标记的干预/备选策略；现有路径不能通过Stage A full-cycle门槛。
