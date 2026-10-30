import os
import re
import glob
import asyncio
from datetime import date, datetime

import numpy as np
import pandas as pd

from StressTester import BaseConfig, OpenAIStressTester


class MdGenerator(BaseConfig):
    """汇总多次压力测试结果，生成 Markdown 结果报告。

    数据来源与 StressTester 当前落盘格式对齐：
        目录: results/outputs/{target}/{model_abbr}/{ISO日期}/{测试类型}/第{N}次/
        文件: {并发数}并发_{YYYYMMDD_HHMMSS}.xlsx
        结构: 第 1 行元信息(合并单元格) / 第 2 行列名 / 第 3 行起请求级明细
    同时兼容改造前的历史文件（首行即列名、旧列名、{并发数}.xlsx 命名）。
    """

    RPS_UNIT = 's'  # 吞吐量单位: Req/{RPS_UNIT}

    def __init__(self, target, is_report: bool = True, model_abbr: str = 'DSV3.1'):
        """
        Args:
            is_report (bool): True  → 正式报告，仅核心指标(成功率/端到端延迟/首Token时间)
                              False → 完整技术明细，追加 TPS/单Token时间/输出Token数/吞吐量
        """
        super().__init__(target)
        self.is_report = is_report
        self.model_abbr = model_abbr

    def read_template(self, md_head_path=None):
        """读取 Markdown 模板"""
        if md_head_path is None:
            md_head_path = os.path.join(self.base_path, 'results', 'mdfiles', 'template.md')
        with open(md_head_path, 'r', encoding='utf-8') as f:
            return f.read()

    # ------------------------------------------------------------------ #
    # 结果文件发现与读取（对齐 StressTester 的新落盘格式）
    # ------------------------------------------------------------------ #
    @property
    def outputs_dir(self) -> str:
        """当前 target + model_abbr 的结果根目录"""
        return os.path.join(self.base_path, 'results', 'outputs',
                            self.target, self.model_abbr)

    def find_result_file(self, test_type: str, concurrency: int, nums_i: int):
        """
        glob 匹配第 nums_i+1 次测试中指定并发数的结果文件。

        目录为 ISO 日期、文件名带开始时刻，无法直接拼接路径，故用递归 glob；
        同一轮次出现多个匹配(如跨天重跑)时取修改时间最新的一个。
        """
        run_dir = os.path.join(self.outputs_dir, '**', test_type, f'第{nums_i + 1}次')

        # 新命名: {并发数}并发_{日期}_{时刻}.xlsx
        matches = glob.glob(os.path.join(run_dir, f'{concurrency}并发_*.xlsx'),
                            recursive=True)
        if not matches:
            # 旧命名: {并发数}.xlsx
            matches = glob.glob(os.path.join(run_dir, f'{concurrency}.xlsx'),
                                recursive=True)
        if not matches:
            return None
        return max(matches, key=os.path.getmtime)

    def load_detail(self, result_path: str):
        """
        读取明细表，兼容两种表头布局：
            新格式: 第 1 行元信息(合并单元格)，第 2 行列名
            旧格式: 第 1 行即列名

        Returns:
            (pd.DataFrame, dict): 明细表与元信息(含成功率/吞吐量等)
        """
        raw = pd.read_excel(result_path, header=None)

        # 在前 3 行内定位列名行(含端到端时间列)，其下即为数据
        header_idx = 0
        for idx in range(min(3, len(raw))):
            row_values = {str(v) for v in raw.iloc[idx]}
            if self.COL_TRT in row_values or '端到端时间' in row_values:
                header_idx = idx
                break

        meta = self.parse_meta(raw.iloc[0, 0]) if header_idx > 0 else {}
        df = raw.iloc[header_idx + 1:].reset_index(drop=True)
        df.columns = [str(c) for c in raw.iloc[header_idx]]
        df.rename(columns=self.legacy_column_map(), inplace=True)
        return df, meta

    def legacy_column_map(self) -> dict:
        """旧列名 → BaseConfig 新列名，兼容历史结果文件"""
        return {
            '端到端时间': self.COL_TRT,
            '首Token时间': self.COL_TFT,
            '单Token时间': self.STAT_TPT,
        }

    @staticmethod
    def parse_meta(meta_line) -> dict:
        """解析元信息行 'k:v | k:v | ...' 为字典"""
        meta = {}
        for part in str(meta_line).split('|'):
            if ':' not in part:
                continue
            key, value = part.split(':', 1)
            meta[key.strip()] = value.strip()
        return meta

    @staticmethod
    def to_numeric(df: pd.DataFrame, column: str) -> pd.Series:
        """取数值列；列不存在时返回空 Series"""
        if column not in df.columns:
            return pd.Series(dtype='float64')
        return pd.to_numeric(df[column], errors='coerce')

    @staticmethod
    def meta_float(meta: dict, key: str):
        """从元信息提取数值，如 '吞吐量: 12.34 req/s' → 12.34"""
        raw_value = meta.get(key)
        if raw_value is None:
            return None
        match = re.search(r'\d+(?:\.\d+)?', str(raw_value))
        return float(match.group()) if match else None

    # ------------------------------------------------------------------ #
    # 单次测试结果统计（口径与 build_stats_df 一致）
    # ------------------------------------------------------------------ #
    def summarize_run(self, df: pd.DataFrame, meta: dict, concurrency: int) -> dict:
        """把单次测试的明细表汇总为指标字典"""
        trt = self.to_numeric(df, self.COL_TRT)
        tft = self.to_numeric(df, self.COL_TFT)
        tps = self.to_numeric(df, self.COL_TPS)
        tokens = self.to_numeric(df, self.COL_TOKEN)

        # 成功率: 优先「是否成功」列，回退 ERROR 为空，再回退 TRT 非空
        if self.COL_SUCCESS in df.columns:
            flags = df[self.COL_SUCCESS].astype(str).str.strip().str.lower()
            success_rate = flags.isin(['true', '1', '1.0']).mean() if len(df) else 0.0
        elif self.COL_ERROR in df.columns:
            success_rate = df[self.COL_ERROR].isnull().mean() if len(df) else 0.0
        else:
            success_rate = trt.notnull().mean() if len(df) else 0.0

        # 单Token时间: 新明细表已无该列(仅统计表保留)，由 (TRT-TFT)/(tokens-1) 派生；
        # 历史文件仍带该列时直接沿用
        if self.STAT_TPT in df.columns:
            tpt = self.to_numeric(df, self.STAT_TPT)
        else:
            denom = (tokens - 1).where(tokens > 1, tokens).replace(0, np.nan)
            tpt = ((trt - tft) / denom).dropna()

        # 吞吐量: 优先元信息中的墙钟时间口径，回退 Little's Law
        rps = self.meta_float(meta, '吞吐量')
        if rps is None and trt.count() and trt.mean() > 0:
            rps = concurrency / trt.mean() * 1000

        return {
            'success_rate': float(success_rate),
            'trt_avg': trt.mean(),
            'trt_p95': trt.quantile(0.95),
            'trt_p99': trt.quantile(0.99),
            'tft_avg': tft.mean(),
            'tft_p95': tft.quantile(0.95),
            'tft_p99': tft.quantile(0.99),
            'tps_avg': tps.mean(),
            'tps_min': tps.min(),
            'token_avg': tokens.mean(),
            'tpt_avg': tpt.mean(),
            'rps': rps,
        }

    # ------------------------------------------------------------------ #
    # 多次测试汇总
    # ------------------------------------------------------------------ #
    def collect_test_results(self, total_nums, test_type: str, concurrency: int) -> list:
        """收集特定并发级别的多次测试结果"""
        test_results = []
        for nums_i in range(total_nums):
            result_path = self.find_result_file(test_type, concurrency, nums_i)
            if result_path is None:
                print(f'未找到[{test_type}-{concurrency}并发-第{nums_i + 1}次]的结果文件')
                continue

            df, meta = self.load_detail(result_path)
            test_results.append(self.summarize_run(df, meta, concurrency))

        return test_results

    def calculate_result_ranges(self, test_results: list) -> tuple:
        """计算多次测试结果的范围"""
        rates = [r['success_rate'] for r in test_results]
        trt_avgs = [r['trt_avg'] for r in test_results]
        tft_avgs = [r['tft_avg'] for r in test_results]
        trt_p95s = [r['trt_p95'] for r in test_results]
        trt_p99s = [r['trt_p99'] for r in test_results]
        tft_p95s = [r['tft_p95'] for r in test_results]
        tft_p99s = [r['tft_p99'] for r in test_results]
        tps_avgs = [r['tps_avg'] for r in test_results]
        tps_mins = [r['tps_min'] for r in test_results]
        per_token_times = [r['tpt_avg'] for r in test_results]
        token_avgs = [r['token_avg'] for r in test_results]
        rps_avgs = [r['rps'] for r in test_results if r['rps'] is not None]

        # 计算成功率范围
        min_rate, max_rate = min(rates), max(rates)
        rate_range = f"{min_rate:.1%}~{max_rate:.1%}" if min_rate != max_rate else f"{min_rate:.1%}"

        # 计算TRT范围
        trt_range = self.merge_format_str(
            avg=self.format_time_range(min(trt_avgs), max(trt_avgs)),
            p95=self.format_time_range(min(trt_p95s), max(trt_p95s)),
            p99=self.format_time_range(min(trt_p99s), max(trt_p99s)))

        # 计算TFT范围
        tft_range = self.merge_format_str(
            avg=self.format_time_range(min(tft_avgs), max(tft_avgs)),
            p95=self.format_time_range(min(tft_p95s), max(tft_p95s)),
            p99=self.format_time_range(min(tft_p99s), max(tft_p99s)))

        # 计算TPS范围
        tps_range = self.merge_format_str(
            avg=self.format_data_range(tps_avgs),
            min_tps=self.format_data_range(tps_mins))

        # 计算其他指标的范围
        per_token_time_range = self.format_data_range(per_token_times)
        token_num_range = self.format_data_range(token_avgs)
        rps_range = self.format_data_range(rps_avgs)

        return rate_range, trt_range, tft_range, tps_range, per_token_time_range, token_num_range, rps_range

    def collect_concurrency_data(self, total_nums, test_type: str) -> dict:
        """收集各并发级别的测试数据"""
        report_data = {
            'rate_values': [], 'trt_values': [], 'tft_values': [], 'tps_values': [],
            'per_token_times': [], 'token_values': [], 'rps_values': [],
        }
        for concurrency in self.concurrencies:
            test_results = self.collect_test_results(total_nums, test_type, concurrency)

            if test_results:
                rate_range, trt_range, tft_range, tps_range, per_token_times, token_values, rps_avg = \
                    self.calculate_result_ranges(test_results)
                report_data['rate_values'].append(rate_range)
                report_data['trt_values'].append(trt_range)
                report_data['tft_values'].append(tft_range)
                report_data['tps_values'].append(tps_range)
                report_data['per_token_times'].append(per_token_times)
                report_data['token_values'].append(token_values)
                report_data['rps_values'].append(rps_avg)
            else:
                # 如果没有数据，填充默认值
                for key in report_data:
                    report_data[key].append("N/A")

        return report_data

    # ------------------------------------------------------------------ #
    # 表格与报告生成
    # ------------------------------------------------------------------ #
    def generate_merge_table(self, report_data: dict) -> str:
        """生成合并结果表格"""
        concurrency_headers = "|".join(str(c) for c in self.concurrencies)

        md_content = f"|并发数量|{concurrency_headers}|\n"
        md_content += "|" + ":------:|" * (len(self.concurrencies) + 1) + "\n"
        md_content += f"|请求成功率|{'|'.join(report_data['rate_values'])}|\n"
        md_content += f"|端到端延迟|{'|'.join(report_data['trt_values'])}|\n"
        md_content += f"|首Token时间|{'|'.join(report_data['tft_values'])}|\n"
        if not self.is_report:
            # 完整技术明细: 追加 TPS / 单Token时间 / 输出Token数 / 吞吐量
            md_content += f"|平均每秒token数|{'|'.join(report_data['tps_values'])}|\n"
            md_content += f"|单token生成时间<br>(ms)|{'|'.join(report_data['per_token_times'])}|\n"
            md_content += f"|输出Token数<br>(Tokens)|{'|'.join(report_data['token_values'])}|\n"
            md_content += f"|吞吐量<br>(Req/{self.RPS_UNIT})|{'|'.join(report_data['rps_values'])}|\n"

        return md_content

    def run(self, total_nums):
        # 保存为Markdown结果，多次测试分别保存
        self.merge_run(total_nums)

    def merge_run(self, total_nums):
        """汇总多次压力测试结果并生成 Markdown 报告"""
        now = datetime.now()
        md_dir = os.path.join(self.base_path, 'results', 'mdfiles',
                              f'{self.month}月', f'{self.month}-{self.day}')
        os.makedirs(md_dir, exist_ok=True)

        # 文件名带生成时刻，避免同一天多次生成报告时互相覆盖
        md_path = os.path.join(
            md_dir,
            f'{self.build_title(total_nums, with_day=False)}'
            f'_{now.strftime("%Y%m%d_%H%M%S")}.md')

        md_output = self.generate_merge_report(total_nums)

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_output)
        print(f"结果已保存到: {os.path.relpath(md_path)}")


    def build_title(self, total_nums: int, with_day: bool = True) -> str:
        """
        报告标题: 压力测试-{target}-{model_abbr}_{测试日期}({月}-{日})-{N}次测试结果汇总

        with_day=False 时省略 ({月}-{日})，用于文件名(与参考文档一致)。
        测试日期取自结果文件名内嵌的 {并发数}并发_{YYYYMMDD_HHMMSS}，
        取不到时回退最新结果文件时间，再取不到时用当天。
        """
        day_part = f'({self.month}-{self.day})' if with_day else ''
        return (f'压力测试-{self.target}-{self.model_abbr}_'
                f'{self.detect_test_date()}{day_part}-{total_nums}次测试结果汇总')

    def detect_test_date(self) -> str:
        """从结果文件中提取测试日期(YYYYMMDD)"""
        dates = set()
        for path in glob.glob(os.path.join(self.outputs_dir, '**', '*并发_*.xlsx'),
                              recursive=True):
            match = re.search(r'并发_(\d{8})_', os.path.basename(path))
            if match:
                dates.add(match.group(1))
        if dates:
            return max(dates)

        all_files = glob.glob(os.path.join(self.outputs_dir, '**', '*.xlsx'),
                              recursive=True)
        if all_files:
            newest = max(all_files, key=os.path.getmtime)
            return datetime.fromtimestamp(os.path.getmtime(newest)).strftime('%Y%m%d')

        return date.today().strftime('%Y%m%d')

    def generate_merge_report(self, total_nums) -> str:
        """生成合并报告的内容"""
        md_report = f'# {self.build_title(total_nums)}\n\n{self.read_template()}\n'

        for type_dict in self.type_configs.values():
            md_report += f"### {type_dict['log']}\n"

            # 收集各并发级别的数据
            report_data = self.collect_concurrency_data(total_nums, type_dict["log"])

            # 生成表格
            md_report += self.generate_merge_table(report_data)

        return md_report

    # ------------------------------------------------------------------ #
    # 格式化
    # ------------------------------------------------------------------ #
    def format_data_range(self, values) -> str:
        """格式化数据范围；无有效值时返回 N/A"""
        valid = [v for v in values if v is not None and pd.notna(v)]
        if not valid:
            return 'N/A'
        return (f"{min(valid):.1f}~{max(valid):.1f}"
                if min(valid) != max(valid) else f"{min(valid):.1f}")

    def format_time_range(self, min_value: float, max_value: float) -> str:
        """格式化时间范围；无有效值时返回 N/A"""
        if pd.isna(min_value) or pd.isna(max_value):
            return 'N/A'
        min_formatted = self.format_time_value(min_value)
        max_formatted = self.format_time_value(max_value)

        return f"{min_formatted}~{max_formatted}" if min_formatted != max_formatted else min_formatted


if __name__ == '__main__':
    # —— 运行参数 ——
    target = 'test'            # 目标配置(configs/envs/{target}.env)
    model_abbr = 'Q2.5Ins0.5B'   # 模型缩写，需与 results/outputs 下的目录名一致
    is_report = True             # True: 正式报告(3行核心指标)；False: 完整技术明细(7行)
    test_times = 1               # 汇总最近 N 次压测的结果
    run_stress = False           # True: 先跑压测再生成报告；False: 仅用已有结果生成报告

    if run_stress:
        tester = OpenAIStressTester(target, model_abbr)
        # 测试类型/并发数默认取自 configs/json/test_type.json 与 BaseConfig，
        # 如需临时缩减范围，取消下面注释并修改:
        # tester.type_configs = {'short_in_short_out': {'log': '短输入短输出', 'max_tokens_num': 20}}
        # tester.concurrencies = [50]
        asyncio.run(tester.run(test_times))

    generator = MdGenerator(target, is_report=is_report, model_abbr=model_abbr)
    generator.run(test_times)
