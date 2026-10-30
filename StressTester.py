"""
大模型推理服务压力测试工具。

输出结构（每个并发一个 xlsx，单 sheet）:
    第1行: 元信息（合并单元格，使单个文件脱离目录后仍可自解释）
    第2行: 列名
    第3行起: 请求级明细

输出路径:
    results/outputs/{target}/{model_abbr}/{ISO日期}/{测试类型}/{第N次}/{并发数}并发_{日期}_{时刻}.xlsx
    例: results/outputs/autodl/Q2.5Ins0.5B/2026-10-23/短输入长输出/第1次/50并发_20261023_101542.xlsx
"""

import os
import time
import json
import asyncio

import numpy as np
import pandas as pd

from datetime import date, datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm as async_tqdm


# —— 全局默认参数 ——
DEFAULT_TEMPERATURE = 0.6
DEFAULT_COLUMN_WIDTH = 18


class BaseConfig:
    """
    配置型父类，包含共用的配置、列常量和统计工具方法。
    """

    # —— 明细表列名（唯一来源，全链路引用，避免硬编码漂移） ——
    COL_SUCCESS = '是否成功'
    COL_TRT     = '端到端时间(ms)'
    COL_TFT     = '首Token时间(ms)'
    COL_TPS     = '每秒传输Token数'
    COL_TOKEN   = '输出Token数'
    COL_CONTENT = '实际生成内容'
    COL_ERROR   = 'ERROR'

    # —— 仅出现在统计表中的指标名（明细表无对应列） ——
    STAT_TPT  = '单Token时间(ms)'
    STAT_RATE = '请求成功率'
    STAT_RPS  = '吞吐量(RPS)'
    STAT_WALL = '批次墙钟时间'

    # —— 统计表列顺序 ——
    STAT_COLUMNS = ['样本数', '平均值', '最小值', 'p50', 'p90', 'p95', 'p99', '最大值', '标准差']

    # —— 明细表中需保留两位小数的数值列 ——
    ROUND_COLUMNS = (COL_TRT, COL_TFT, COL_TPS)

    def __init__(self, target):
        """
        初始化基础配置

        Args:
            target (str): 目标配置名称
        """
        self.target = target
        self.month = date.today().month
        self.day = date.today().day

        # 基础通用参数
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.load_env()

        # 测试类型参数
        self.type_configs = self.get_type_configs()
        self.concurrencies = [50, 100, 150, 200, 250]

    def load_env(self):
        """加载环境变量"""
        load_dotenv(os.path.join(self.base_path, 'configs', 'envs', f'{self.target}.env'))

    def get_type_configs(self):
        """加载测试类型配置"""
        with open(os.path.join(self.base_path, 'configs', 'json', 'test_type.json'), 'r', encoding='utf-8') as f:
            return json.load(f)

    def format_str(self, avg, p95=None, p99=None):
        if p95 and p99:
            return f"平均：{self.format_time_value(avg)}<br>p95：{self.format_time_value(p95)}<br>p99：{self.format_time_value(p99)}"
        else:
            return f"平均：{self.format_time_value(avg)}"

    def merge_format_str(self, avg, p95=None, p99=None, min_tps=None):
        if p95 and p99:
            return f"平均：{avg}<br>p95：{p95}<br>p99：{p99}"
        elif min_tps:
            return f"平均：{avg}<br>最小：{min_tps}"
        else:
            return f"平均：{avg}"

    @staticmethod
    def format_time_value(value):
        """格式化时间值"""
        return f"{value:.0f}ms" if value < 1000 else f"{value/1000:.1f}s"

    # ------------------------------------------------------------------ #
    # 统计表构建：日志与 Markdown 报告的唯一统计口径来源
    # ------------------------------------------------------------------ #
    def build_stats_df(self, detail_df, per_token_times, success_rate, rps, wall_clock):
        """
        构建统计表（行=指标，列=统计项）。

        Args:
            detail_df (pd.DataFrame): 明细表
            per_token_times (pd.Series): 单Token时间，明细表已无此列，独立传入
            success_rate (float): 请求成功率 0~1
            rps (float): 吞吐量(req/s)
            wall_clock (float): 批次墙钟时间(s)，RPS 的分母，保留用于反推验算

        Returns:
            pd.DataFrame: 统计表
        """
        rows = [
            self._scalar_row(self.STAT_RATE, '%', len(detail_df), round(success_rate * 100, 2)),
            self._scalar_row(self.STAT_RPS, 'req/s', len(detail_df), round(rps, 2)),
            self._scalar_row(self.STAT_WALL, 's', len(detail_df), round(wall_clock, 2)),
        ]
        rows += [
            self._variable_row(self.COL_TRT,   'ms',       detail_df[self.COL_TRT]),
            self._variable_row(self.COL_TFT,   'ms',       detail_df[self.COL_TFT]),
            self._variable_row(self.COL_TPS,   'tokens/s', detail_df[self.COL_TPS]),
            self._variable_row(self.COL_TOKEN, 'tokens',   detail_df[self.COL_TOKEN]),
            self._variable_row(self.STAT_TPT,  'ms',       per_token_times),
        ]
        return pd.DataFrame(rows, columns=['指标', '单位'] + self.STAT_COLUMNS)

    @classmethod
    def _scalar_row(cls, name, unit, sample_num, value):
        """
        用于成功率、吞吐量、墙钟时间这类对整批请求只有一个取值、不存在分布的指标。
        """
        row = {'指标': name, '单位': unit, '样本数': sample_num}
        row.update({col: None for col in cls.STAT_COLUMNS if col != '样本数'})
        row['平均值'] = value

        return row

    @classmethod
    def _variable_row(cls, name, unit, series):
        """
        分布指标行：输出样本数 + 平均值 / 最小值 / p50 / p90 / p95 / p99 / 最大值 / 标准差。
        """
        values = pd.Series(series).dropna()
        row = cls._scalar_row(name, unit, int(values.size), None)
        if values.empty:
            return row

        row.update({
            '平均值': round(float(values.mean()), 2),
            '最小值': round(float(values.min()), 2),
            'p50':    round(float(values.quantile(.50)), 2),
            'p90':    round(float(values.quantile(.90)), 2),
            'p95':    round(float(values.quantile(.95)), 2),
            'p99':    round(float(values.quantile(.99)), 2),
            '最大值': round(float(values.max()), 2),
            '标准差': round(float(values.std()), 2),
        })
        return row


class OpenAIStressTester(BaseConfig):
    def __init__(self, target, model_abbr: str = 'DSV3.1'):
        """
        OpenAI兼容节点压测类。

        Args:
            target (str): target。
        """
        super().__init__(target)
        self.model_abbr = model_abbr
        self.api_configs_init()
        self.model_configs_init()

    def api_configs_init(self):
        """
        初始化客户端配置。
        """
        # API配置
        self.OPENAI_URL = os.getenv('OPENAI_URL')
        self.API_KEY = os.getenv('API_KEY')

        # 异步OpenAI客户端
        self.async_client = AsyncOpenAI(base_url=self.OPENAI_URL, api_key=self.API_KEY)

    def model_configs_init(self):
        """
        初始化模型相关配置。
        """
        # 模型配置
        self.MODEL_NAME = os.getenv('MODEL_NAME') or ''
        self.TOKENIZER_PATH = os.getenv('TOKENIZER_PATH')
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_PATH)

        # 加载提示词
        prompt_path = os.path.join(self.base_path, 'configs', 'json', 'prompts.json')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompts_dict = json.load(f)

    async def fetch(self, test_type, is_stream=False):
        """
        发起单次请求并采集时间指标。

        Returns:
            成功: (token数, 请求开始时刻, 首Token时刻, 结束时刻, 单Token时间, 生成内容)
            失败: {'type': 异常类名, 'message': 异常信息, 'status_code': HTTP状态码}
        """
        prompt: str = self.prompts_dict[test_type][0]
        max_tokens_num = self.type_configs[test_type]["max_tokens_num"]

        req_st_time = time.perf_counter()
        first_token_time = None
        full_content = ""
        time_stamps = []

        try:
            response = await self.async_client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}],
                max_completion_tokens=max_tokens_num,
                temperature=DEFAULT_TEMPERATURE,
                stream=is_stream
            )
            if is_stream:
                async for chunk in response:
                    answer_chunk = chunk.choices[0].delta.content
                    if answer_chunk:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        time_stamps.append(time.perf_counter())
                        full_content += answer_chunk
            else:
                full_content = response.choices[0].message.content.strip()
                first_token_time = time.perf_counter()
            end_time = time.perf_counter()

            # 虽然第一个为特殊token，但也被计入总长度
            token_count = len(self.tokenizer.encode(full_content))

            per_token_times = []
            if len(time_stamps) > 1:
                per_token_times = [time_stamps[i + 1] - time_stamps[i]
                                   for i in range(len(time_stamps) - 1)]
            else:
                per_token_times.append(time_stamps[0] - req_st_time)
            per_token_time = np.mean(per_token_times)

            return token_count, req_st_time, first_token_time, end_time, per_token_time, full_content

        except Exception as e:
            status_code = getattr(e, 'status_code', None)
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'status_code': status_code,
            }

            # 输出详细的错误信息
            print(f"{'='*60}")
            print(f"❌ [错误详情]: {str(e)}")
            print(f"   测试类型: {self.type_configs[test_type]['log']}")
            print(f"   模型: {self.MODEL_NAME}")
            print(f"   异常类型: {type(e).__name__}")
            if status_code is not None:
                # 明确一下这个判定的作用，应该是便于直接确认网络问题
                print(f"   HTTP状态码: {status_code}")
            print(f"{'='*60}\n")

            return error_info

    async def fetch_with_pbar(self, pbar, test_type):
        result = await self.fetch(test_type, is_stream=True)
        if pbar:  # 确保进度条存在
            pbar.update(1)

        return result

    # ------------------------------------------------------------------ #
    # 并发编排：发起请求 → 解析 → 统计 → 打印 → 落盘
    # ------------------------------------------------------------------ #
    async def _dispatch_requests(self, test_type, concurrency, nums_i, with_pbar=True):
        """
        按给定并发数发起全部请求。

        Args:
            with_pbar (bool): 是否显示进度条（预热阶段不显示）
        """
        if not with_pbar:
            tasks = [self.fetch(test_type, is_stream=True) for _ in range(concurrency)]
            return await asyncio.gather(*tasks, return_exceptions=True)

        pbar = None
        try:
            # 建进度条
            pbar = async_tqdm(
                total=concurrency,
                desc=f"{self.type_configs[test_type]['log']}-{concurrency}并发-第{nums_i}次测试",
                unit="req", unit_scale=False, ncols=80, leave=True
            )
            # 建任务
            tasks = [self.fetch_with_pbar(pbar, test_type) for _ in range(concurrency)]
            return await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if pbar:
                pbar.close()

    def _parse_responses(self, responses):
        """
        把原始响应解析为明细表与派生数据。

        asyncio.gather(return_exceptions=True) 使 responses 存在三种形态，
        这里做显式三分支分发，避免"不是 dict 就一定是 6 元组"的隐式假设。

        Returns:
            (detail_df, per_token_times, req_starts, req_ends)
        """
        records = {
            self.COL_TRT: [], self.COL_TFT: [], self.COL_TPS: [],
            self.COL_TOKEN: [], self.COL_CONTENT: [], self.COL_ERROR: [],
        }
        success_flags, per_token_times = [], []
        req_starts, req_ends = [], []

        for r in responses:
            if isinstance(r, dict):
                # fetch 内部捕获的异常：返回错误详情字典
                self._append_failed_record(records, success_flags, per_token_times,
                                           req_starts, req_ends, error=r['type'])
            elif isinstance(r, BaseException):
                # 协程内抛出但未被 fetch 捕获的异常（如 prompts_dict 的 KeyError）
                self._append_failed_record(records, success_flags, per_token_times,
                                           req_starts, req_ends, error=type(r).__name__)
            else:
                # 正常响应：6 元组
                self._append_success_record(records, success_flags, per_token_times,
                                            req_starts, req_ends, response=r)

        detail_df = pd.DataFrame({self.COL_SUCCESS: success_flags, **records})

        # 数值列统一保留两位小数：落盘与统计明细共用同一份数据，避免口径差
        detail_df[list(self.ROUND_COLUMNS)] = detail_df[list(self.ROUND_COLUMNS)].round(2)
        per_token_series = pd.Series(per_token_times, dtype='float64').round(2)

        return detail_df, per_token_series, req_starts, req_ends

    def _append_failed_record(self, records, success_flags, per_token_times,
                              req_starts, req_ends, error):
        """失败请求：度量列置空，仅记录错误类名。"""
        for col in (self.COL_TRT, self.COL_TFT, self.COL_TPS,
                    self.COL_TOKEN, self.COL_CONTENT):
            records[col].append(None)
        records[self.COL_ERROR].append(error)

        success_flags.append(False)
        per_token_times.append(None)
        req_starts.append(None)
        req_ends.append(None)

    def _append_success_record(self, records, success_flags, per_token_times,
                               req_starts, req_ends, response):
        """成功请求：计算 TRT / TFT / TPS / 单Token时间。"""
        token_count, req_st_time, first_token_time, end_time, token_time, full_content = response
        elapsed = end_time - req_st_time

        records[self.COL_TRT].append(elapsed * 1000)
        records[self.COL_TFT].append((first_token_time - req_st_time) * 1000)
        records[self.COL_TPS].append(token_count / elapsed if elapsed > 0 else 0)
        records[self.COL_TOKEN].append(token_count)
        records[self.COL_CONTENT].append(full_content)
        records[self.COL_ERROR].append(None)

        success_flags.append(True)
        per_token_times.append(token_time * 1000)
        req_starts.append(req_st_time)
        req_ends.append(end_time)

    @staticmethod
    def _calc_wall_clock(req_starts, req_ends):
        """
        批次墙钟时间(秒) = 最后一个请求返回时刻 − 第一个请求发出时刻。

        这是黑盒实测吞吐的分母：相比 Little's Law（并发数 ÷ 平均延迟），
        它不受"尾部请求在途并发下降"的影响，高并发长尾场景下不会高估吞吐。
        """
        starts = [t for t in req_starts if t is not None]
        ends = [t for t in req_ends if t is not None]
        if not starts or not ends:
            return 0.0
        return max(ends) - min(starts)

    @staticmethod
    def _calc_rps(success_num, wall_clock):
        """吞吐量(req/s) = 成功请求数 ÷ 批次墙钟时间。"""
        return success_num / wall_clock if wall_clock > 0 else 0.0

    def print_logs(self, stats_df, test_type, concurrency):
        """
        基于统计表打印本轮压测摘要（口径与统计表完全一致）。

        Args:
            stats_df (pd.DataFrame): build_stats_df 的输出
        """
        table = stats_df.set_index('指标')

        def value_of(metric, column='平均值'):
            try:
                return table.loc[metric, column]
            except KeyError:
                return None

        def time_of(metric, column='平均值'):
            val = value_of(metric, column)
            return self.format_time_value(val) if pd.notna(val) else 'N/A'

        def num_of(metric, column='平均值', digits=1):
            val = value_of(metric, column)
            return f"{val:.{digits}f}" if pd.notna(val) else 'N/A'

        rate = value_of(self.STAT_RATE)
        print(f"{self.type_configs[test_type]['log']}-{concurrency}并发测试:")
        print(f"请求成功率: {rate:.1f}%" if pd.notna(rate) else "请求成功率: N/A")

        summary_items = [('平均值', '平均'), ('最小值', '最小'),
                         ('p50', 'p50'), ('p95', 'p95'),
                         ('p99', 'p99'), ('最大值', '最大')]
        for label, metric in [('端到端时间(TRT)', self.COL_TRT),
                              ('首Token时间(TFT)', self.COL_TFT)]:
            print(f"{label}:")
            for column, name in summary_items:
                print(f"    {name}: {time_of(metric, column)}")

        print(f"平均每秒传输Token数(TPS): {num_of(self.COL_TPS)}")
        print(f"平均输出Token数: {num_of(self.COL_TOKEN)}")
        print(f"平均单Token时间: {num_of(self.STAT_TPT)}ms")
        print(f"批次墙钟时间: {num_of(self.STAT_WALL, digits=2)}s")
        print(f"吞吐量(RPS): {num_of(self.STAT_RPS, digits=2)} req/s")
        print("-" * 50)

    def _build_meta_line(self, test_type, nums_i, success_rate, rps) -> str:
        """
        生成写在明细表第 1 行的元信息，使单个 xlsx 脱离目录后仍可自解释。

        """
        meta = {
            '目标':          self.target,
            '模型':          f'{self.model_abbr}',
            '测试类型':      f'{self.type_configs[test_type]["log"]}',
            '轮次':          f'第{nums_i}次',
            '最大输出Token': self.type_configs[test_type]['max_tokens_num'],
            '成功率':        f'{success_rate:.1%}',
            '吞吐量':        f'{rps:.2f} req/s',
        }
        return ' | '.join(f'{k}:{v}' for k, v in meta.items())

    def _build_output_path(self, test_type, concurrency, nums_i, start_dt) -> str:
        """
        构造输出路径。

        目录用 ISO 日期，文件名带并发数与开始时刻
        """
        output_dir = os.path.join(
            self.base_path, 'results', 'outputs', self.target, self.model_abbr,
            start_dt.strftime('%Y-%m-%d'),
            self.type_configs[test_type]['log'], f'第{nums_i}次',
        )
        os.makedirs(output_dir, exist_ok=True)

        filename = f'{concurrency}并发_{start_dt.strftime("%Y%m%d_%H%M%S")}.xlsx'
        return os.path.join(output_dir, filename)

    def _save_result_excel(self, test_type, concurrency, nums_i,
                           detail_df, success_rate, rps, start_dt):
        """
        落盘明细表：第 1 行元信息 / 第 2 行列名 / 第 3 行起数据。
        """
        output_path = self._build_output_path(test_type, concurrency, nums_i, start_dt)

        meta_line = self._build_meta_line(test_type, nums_i, success_rate, rps)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # startrow=1：把第 1 行留给元信息
            detail_df.to_excel(writer, sheet_name='明细', index=False, startrow=1)

            worksheet = writer.sheets['明细']
            worksheet.cell(row=1, column=1, value=meta_line)
            worksheet.merge_cells(start_row=1, start_column=1,
                                  end_row=1, end_column=max(len(detail_df.columns), 2))
            worksheet.freeze_panes = 'A3'   # 冻结元信息行与列名行

            for col_idx in range(1, len(detail_df.columns) + 1):
                letter = worksheet.cell(row=2, column=col_idx).column_letter
                worksheet.column_dimensions[letter].width = DEFAULT_COLUMN_WIDTH

        print(f"详细测试结果已保存至: {output_path}")

    async def evaluate(self, test_type, concurrency, nums_i, is_warmup=False):
        """
        执行并发测试并保存结果。

        Args:
            test_type (str): 测试类型。
            concurrency (int): 并发数。
            nums_i (int): 第几次测试。
            is_warmup (bool): 是否为预热阶段（预热只打印、不落盘）。
        """
        # 记录批次开始时刻：目录日期与文件名时间同源，避免跨午夜不一致
        start_dt = datetime.now()

        responses = await self._dispatch_requests(
            test_type, concurrency, nums_i, with_pbar=not is_warmup)

        detail_df, per_token_times, req_starts, req_ends = self._parse_responses(responses)

        # 计算成功率与吞吐量
        success_num = int(detail_df[self.COL_SUCCESS].sum())
        success_rate = success_num / len(detail_df) if len(detail_df) else 0.0
        wall_clock = self._calc_wall_clock(req_starts, req_ends)
        rps = self._calc_rps(success_num, wall_clock)

        # 构建统计表并打印日志
        stats_df = self.build_stats_df(
            detail_df, per_token_times, success_rate, rps, wall_clock)
        self.print_logs(stats_df, test_type, concurrency)

        # 预热阶段在此跳出
        if is_warmup:
            return

        # 6) 落盘
        if success_rate > 0:
            self._save_result_excel(test_type, concurrency, nums_i,
                                    detail_df, success_rate, rps, start_dt)
        else:
            print(f"警告：{test_type}-{concurrency}并发下没有收到任何成功的响应")

    async def run(self, nums):
        for i in range(nums):
            print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<第{i+1}次压测开始>>>>>>>>>>>>>>>>>>>>>>>>')
            for test_type, type_dict in self.type_configs.items():
                print(f"{type_dict['log']}-第{i+1}次压测-[预热阶段]:")
                await self.evaluate(test_type, 2, i+1, is_warmup=True)
                print(f"{type_dict['log']}-第{i+1}次压测-[预热完成]")

                for concurrency in self.concurrencies:
                    print(f"{type_dict['log']}-{concurrency}并发-第{i+1}次[正式压测]开始")
                    await self.evaluate(test_type, concurrency, i + 1)
                    print(f"{type_dict['log']}-{concurrency}并发-第{i+1}次[正式压测]完成")
            print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<第{i+1}次压测完成>>>>>>>>>>>>>>>>>>>>>>>>')


if __name__ == '__main__':
    target = 'test'
    model_abbr = 'Q2.5Ins0.5B'

    wm_tp = 'short_in_short_out'
    test_times = 2
    tester = OpenAIStressTester(target, model_abbr)
    tester.concurrencies = [
        50,
        100,
        150, 
        200, 
        250
    ]
    asyncio.run(tester.run(test_times))
