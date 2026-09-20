
## 结构处理

| 脚本 | 功能 | 注意事项 |
|---|---|---|
| `collect_bond.py` | 读取 POSCAR/CONTCAR，按指定元素对和距离窗口统计周期性键数及平均键长，默认读取 CONTCAR。 | 在顶部 `BOND_LIST` 修改键类型；计数为每晶胞不重复的无向键数，包含周期镜像，不等于单原子配位数。 |
| `collect_contcar.sh` | 从当前目录的子目录中收集 CONTCAR 到 `structure/`，按来源路径命名，重名时追加编号。 | 跳过当前目录的 CONTCAR，以及脚本中列出的 freq、DOS、band、CDD等目录；不会判断结构是否已收敛。 |
| `get_coor_int.py` | 提取 OUTCAR 各离子步的坐标、力模长和能量到 `OUTCAR.pos`，可通过步号参数另存结构，如 `python3 get_coor_int.py 1 10`。 | 需要匹配的 POSCAR；OUTCAR不完整或坐标块与能量记录数量不一致时，能量可能与相邻步骤错配。 |
| `vasp2cif.py` | 将 POSCAR/CONTCAR 转为同名追加 `.cif` 的文件，支持直接坐标、笛卡尔坐标和常见缩放形式。 | 默认扫描当前目录的 POSCAR*、CONTCAR*，两类恰好各一个时只转换CONTCAR；VASP4格式需要同目录POTCAR，CIF不保留Selective Dynamics约束。 |
| `plot_contcar.sh` | 对当前目录的 CONTCAR_* 文件调用VESTA，批量导出侧视图和俯视图PNG。 | 需要VESTA及虚拟显示工具；当前输出命名会截去最后一个点及其后内容，带小数的扫描文件名可能造成图片互相覆盖。 |

## 电荷与功函数

| 脚本 | 功能 | 注意事项 |
|---|---|---|
| `getbader.py` | 调用 `chgsum.pl` 和Bader，以 AECCAR0+AECCAR2 为参考划分CHGCAR，并将逐原子净电荷写入 `bader_charge_summary.csv`。 | 需要POSCAR、POTCAR、CHGCAR、AECCAR0、AECCAR2；净电荷定义为POTCAR的ZVAL减去Bader电子数，正值表示失电子。 |
| `getddec6.py` | 从 `DDEC6_even_tempered_net_atomic_charges.xyz` 的首个原子数据块提取元素和净电荷，写入 `DDEC6_charges_simpleblock.csv`。 | 仅整理已有DDEC6结果，不执行DDEC6电荷计算。 |
| `getCDD_plot_planar_ave.sh` | 调用VASPKIT计算总体系减去吸附物和基底的差分电荷，并输出指定方向的平面平均数据及SVG，方向参数为x/y/z，默认z。 | 需要当前目录、ads/、surf/中的CHGCAR具有相同晶胞、网格和对应原子几何；输出包括 `CHGDIFF.vasp`、`PLANAR_AVERAGE.dat`、`PLANAR_AVERAGE_z.svg`等。 |
| `getwf.sh` | 调用VASPKIT 426计算z方向功函数，将完整日志保存为 `vaspkit_426.log`，并打印功函数结果行。 | 需要LOCPOT和DOSCAR，真空方向应沿z；非对称slab需确认结果对应所关注的表面。 |
| `plot_wf.py` | 读取 `PLANAR_AVERAGE.dat` 绘制电势曲线，结合OUTCAR或手动输入的费米能级标注功函数，默认输出 `wf.svg`。 | 可用 `--vac-range 起点 终点` 指定真空平台；自动识别可能误选倾斜电势或另一侧表面，输入应为电势而非差分电荷的平面平均数据。 |

## 能带与态密度

| 脚本 | 功能 | 注意事项 |
|---|---|---|
| `getband.sh` | 调用VASPKIT 211，从已完成的能带计算中提取能带数据。 | 在能带计算目录运行，并准备匹配的EIGENVAL、KPOINTS、POSCAR及VASPKIT所需辅助文件。 |
| `getEgap.py` | 从EIGENVAL的能量和占据数判断金属性或提取带隙，支持自旋极化及不同占据数归一化方式，并在有隙时打印带边位置。 | 建议保留匹配的OUTCAR供自动识别；部分占据或归一化依据不足时会提示无法确定，带隙仅代表输入k点采样范围。 |
| `plot_band_vbmcbm.py` | 读取BAND.dat绘制能带，结合BAND_GAP、KLABELS及KPOINTS标注VBM/CBM和带隙，输出 `band_plot.svg`。 | 输入需采用一致能量零点；缺少可靠带隙信息或带边无法匹配时仍绘图，但省略带隙标注。 |
| `plot_wannier_band.py` | 对比VASP Line-mode能带与Wannier90插值能带，输出三张SVG及误差、覆盖率诊断报告。 | VASP侧需要POSCAR/KPOINTS/EIGENVAL，Wannier侧需要seed_band.dat和seed_band.gnu，默认从Wannier目录上一级OUTCAR读取费米能级；自旋计算需声明匹配通道，误差应结合未匹配态和覆盖率解释。 |
| `getbandcenter.sh` | 调用VASPKIT 503进行带中心分析，并接收一个参数作为交互流程的最后一项输入。 | 积分窗口固定为−15至15 eV，需按研究目的调整；参数含义及输入顺序应核对本机VASPKIT交互提示。 |
| `get_dos_each_atom.py` | 接收多个原子编号，逐个调用VASPKIT提取PDOS并按编号保存，如 `python3 get_dos_each_atom.py 153 154`。 | 重名输出默认拒绝覆盖，可加 `--overwrite`；运行前核对本机114任务是否适用脚本中的 `114→1→原子编号` 输入顺序。 |
| `plot_PDOS.py` | 将TDOS与按元素规则选择的轨道PDOS绘制到 `DOS.svg`，规则为H取s、过渡金属取d、镧锕系取f、其余元素取p。 | 当前版本对VASPKIT表头中的 `dx2` 列存在漏加问题，d轨道求和前应修正；可用一个位置参数设置纵轴上限。 |
| `plot_TDOS_LDOS.py` | 读取TDOS及各元素PDOS文件的最后一列，将总DOS与元素投影总和绘制到 `DOS.svg`。 | PDOS最后一列应为轨道总和；当前版本不实际翻转正值DW数据，自旋镜像图需输入DW已为负值。 |

## COHP分析

| 脚本 | 功能 | 注意事项 |
|---|---|---|
| `plot_cohp_avg.py` | 读取 `COHPCAR.lobster` 的average条目，自旋求和后绘制−COHP与−ICOHP，输出 `cohp_icohp_avg_smooth.svg`。 | average对应文件中已有的平均条目，不代表任意指定键组；高斯平滑宽度以数据点数为单位。 |
| `plot_cohp_separate.py` | 读取 `COHPCAR.lobster`，为每个非average键条目分别绘制自旋求和的−COHP与−ICOHP，保存到 `cohp_plots/`。 | 文件名使用COHP条目编号，具体原子对需查原始文件；默认对曲线进行轻微高斯平滑。 |

## 振动与分子动力学

| 脚本 | 功能 | 注意事项 |
|---|---|---|
| `getfreq.sh` | 遍历当前目录下一层子目录中的freq/，调用VASPKIT 501计算298 K热修正，并汇总到 `freq.summary`。 | 在批量计算的父目录运行，目录名避免空格；需检查频率质量和虚频，脚本不会自动判断热修正是否适用。 |
| `plot_aimd_energy.py` | 从 `job.out` 提取各步E0并绘制能量随步数的变化，输出 `e0_plot.svg`。 | E0不是包含离子动能等项的完整MD守恒能量；拼接续算日志可能出现步号重新起算。 |
