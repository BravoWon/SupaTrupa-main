from __future__ import annotations
import re
import json
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from pathlib import Path
from jones_framework.core.manifold_bridge import bridge, ConnectionType
from jones_framework.engine.core import Document, DocumentType, ProcessedDocument

class _clI129c(Enum):
    FORM_10K = '10-K'
    FORM_10Q = '10-Q'
    FORM_8K = '8-K'
    FORM_DEF14A = 'DEF 14A'
    FORM_13F = '13-F'
    FORM_4 = 'Form 4'
    FORM_S1 = 'S-1'
    FORM_424B = '424B'
    FORM_6K = '6-K'
    FORM_20F = '20-F'
    UNKNOWN = 'Unknown'

class _c11029d(Enum):
    ITEM_1_01 = 'Entry into Material Agreement'
    ITEM_1_02 = 'Termination of Material Agreement'
    ITEM_1_03 = 'Bankruptcy'
    ITEM_2_01 = 'Acquisition or Disposition of Assets'
    ITEM_2_02 = 'Results of Operations'
    ITEM_2_03 = 'Creation of Obligation'
    ITEM_2_04 = 'Triggering Events'
    ITEM_2_05 = 'Costs Associated with Exit'
    ITEM_2_06 = 'Material Impairments'
    ITEM_3_01 = 'Delisting'
    ITEM_3_02 = 'Unregistered Sales of Equity'
    ITEM_3_03 = 'Material Modification of Rights'
    ITEM_4_01 = 'Changes in Accountant'
    ITEM_4_02 = 'Non-Reliance on Financials'
    ITEM_5_01 = 'Changes in Control'
    ITEM_5_02 = 'Departure of Directors/Officers'
    ITEM_5_03 = 'Amendments to Articles'
    ITEM_5_04 = 'Temporary Suspension of Trading'
    ITEM_5_05 = 'Amendments to Code of Ethics'
    ITEM_5_06 = 'Change in Shell Company Status'
    ITEM_5_07 = 'Shareholder Vote'
    ITEM_5_08 = 'Shareholder Nominations'
    ITEM_7_01 = 'Regulation FD Disclosure'
    ITEM_8_01 = 'Other Events'
    ITEM_9_01 = 'Financial Statements and Exhibits'

@dataclass
class _cI1029E:
    label: str
    value: float
    unit: str = 'USD'
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    is_quarterly: bool = False
    footnotes: List[str] = field(default_factory=list)

@dataclass
class _c0OO29f:
    period_end: date
    is_quarterly: bool
    revenue: Optional[float] = None
    cost_of_revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_expenses: Optional[float] = None
    rd_expense: Optional[float] = None
    sga_expense: Optional[float] = None
    operating_income: Optional[float] = None
    interest_expense: Optional[float] = None
    interest_income: Optional[float] = None
    other_income: Optional[float] = None
    pretax_income: Optional[float] = None
    tax_expense: Optional[float] = None
    net_income: Optional[float] = None
    eps_basic: Optional[float] = None
    eps_diluted: Optional[float] = None
    shares_basic: Optional[float] = None
    shares_diluted: Optional[float] = None
    raw_items: Dict[str, _cI1029E] = field(default_factory=dict)

    @property
    def _flIO2AO(self) -> Optional[float]:
        if self.revenue and self.gross_profit:
            return self.gross_profit / self.revenue
        return None

    @property
    def _fOI02Al(self) -> Optional[float]:
        if self.revenue and self.operating_income:
            return self.operating_income / self.revenue
        return None

    @property
    def _f11O2A2(self) -> Optional[float]:
        if self.revenue and self.net_income:
            return self.net_income / self.revenue
        return None

@dataclass
class _c1112A3:
    period_end: date
    cash: Optional[float] = None
    short_term_investments: Optional[float] = None
    accounts_receivable: Optional[float] = None
    inventory: Optional[float] = None
    other_current_assets: Optional[float] = None
    total_current_assets: Optional[float] = None
    ppe_net: Optional[float] = None
    goodwill: Optional[float] = None
    intangibles: Optional[float] = None
    long_term_investments: Optional[float] = None
    other_assets: Optional[float] = None
    total_assets: Optional[float] = None
    accounts_payable: Optional[float] = None
    short_term_debt: Optional[float] = None
    accrued_liabilities: Optional[float] = None
    deferred_revenue: Optional[float] = None
    other_current_liabilities: Optional[float] = None
    total_current_liabilities: Optional[float] = None
    long_term_debt: Optional[float] = None
    deferred_tax_liabilities: Optional[float] = None
    other_liabilities: Optional[float] = None
    total_liabilities: Optional[float] = None
    common_stock: Optional[float] = None
    retained_earnings: Optional[float] = None
    treasury_stock: Optional[float] = None
    accumulated_other_comprehensive_income: Optional[float] = None
    total_equity: Optional[float] = None
    raw_items: Dict[str, _cI1029E] = field(default_factory=dict)

    @property
    def _f1Ol2A4(self) -> Optional[float]:
        if self.total_current_assets and self.total_current_liabilities:
            return self.total_current_assets / self.total_current_liabilities
        return None

    @property
    def _flll2A5(self) -> Optional[float]:
        total_debt = (self.short_term_debt or 0) + (self.long_term_debt or 0)
        if self.total_equity and total_debt:
            return total_debt / self.total_equity
        return None

@dataclass
class _clOl2A6:
    period_end: date
    is_quarterly: bool
    net_income: Optional[float] = None
    depreciation: Optional[float] = None
    stock_comp: Optional[float] = None
    changes_working_capital: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    capex: Optional[float] = None
    acquisitions: Optional[float] = None
    investments: Optional[float] = None
    investing_cash_flow: Optional[float] = None
    debt_issued: Optional[float] = None
    debt_repaid: Optional[float] = None
    stock_issued: Optional[float] = None
    stock_repurchased: Optional[float] = None
    dividends_paid: Optional[float] = None
    financing_cash_flow: Optional[float] = None
    net_change_cash: Optional[float] = None
    beginning_cash: Optional[float] = None
    ending_cash: Optional[float] = None
    raw_items: Dict[str, _cI1029E] = field(default_factory=dict)

    @property
    def _fl1O2A7(self) -> Optional[float]:
        if self.operating_cash_flow is not None and self.capex is not None:
            return self.operating_cash_flow - abs(self.capex)
        return None

@dataclass
class _cll02A8:
    title: str
    description: str
    category: str
    severity: str
    is_new: bool = False
    is_modified: bool = False
    keywords: List[str] = field(default_factory=list)

@dataclass
class _cl1l2A9:
    name: str
    title: str
    year: int
    base_salary: float = 0
    bonus: float = 0
    stock_awards: float = 0
    option_awards: float = 0
    non_equity_incentive: float = 0
    pension_value: float = 0
    other_compensation: float = 0

    @property
    def _f0O12AA(self) -> float:
        return self.base_salary + self.bonus + self.stock_awards + self.option_awards + self.non_equity_incentive + self.pension_value + self.other_compensation

@dataclass
class _cl002AB:
    filing_type: _clI129c
    cik: str
    company_name: str
    ticker: Optional[str]
    filing_date: date
    period_end: date
    accession_number: str
    income_statements: List[_c0OO29f] = field(default_factory=list)
    balance_sheets: List[_c1112A3] = field(default_factory=list)
    cash_flow_statements: List[_clOl2A6] = field(default_factory=list)
    risk_factors: List[_cll02A8] = field(default_factory=list)
    items_reported: List[_c11029d] = field(default_factory=list)
    executive_compensation: List[_cl1l2A9] = field(default_factory=list)
    mda_summary: str = ''
    mda_sentiment: Dict[str, float] = field(default_factory=dict)
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    sections: Dict[str, str] = field(default_factory=dict)
    word_count: int = 0
    processing_time_ms: float = 0

@bridge(connects_to=['DocumentProcessor', 'JonesEngine', 'LinguisticArbitrageEngine'], connection_types={'DocumentProcessor': ConnectionType.EXTENDS, 'JonesEngine': ConnectionType.USES, 'LinguisticArbitrageEngine': ConnectionType.PRODUCES})
class _cIl12Ac:

    def __init__(self):
        self._10k_sections = {'business': '(?:ITEM\\s*1[.\\s]*BUSINESS)', 'risk_factors': '(?:ITEM\\s*1A[.\\s]*RISK\\s*FACTORS)', 'properties': '(?:ITEM\\s*2[.\\s]*PROPERTIES)', 'legal': '(?:ITEM\\s*3[.\\s]*LEGAL\\s*PROCEEDINGS)', 'mda': '(?:ITEM\\s*7[.\\s]*MANAGEMENT)', 'financials': '(?:ITEM\\s*8[.\\s]*FINANCIAL\\s*STATEMENTS)', 'controls': '(?:ITEM\\s*9A[.\\s]*CONTROLS)'}
        self._income_patterns = {'revenue': ['(?:total\\s+)?(?:net\\s+)?revenue', '(?:total\\s+)?(?:net\\s+)?sales'], 'cost_of_revenue': ['cost\\s+of\\s+(?:revenue|sales|goods)', 'cost\\s+of\\s+products'], 'gross_profit': ['gross\\s+profit', 'gross\\s+margin'], 'operating_income': ['(?:income|loss)\\s+from\\s+operations', 'operating\\s+(?:income|loss)'], 'net_income': ['net\\s+(?:income|loss)', 'net\\s+earnings']}
        self._ticker_map: Dict[str, str] = {}

    async def _fl002Ad(self, _f0Il2AE: Document) -> _cl002AB:
        import time
        start_time = time.time()
        if isinstance(_f0Il2AE.content, bytes):
            text = _f0Il2AE.content.decode('utf-8', errors='ignore')
        else:
            text = _f0Il2AE.content
        text = self._clean_markup(text)
        filing_type = self._detect_filing_type(text)
        header = self._parse_header(text)
        result = _cl002AB(filing_type=filing_type, cik=header.get('cik', ''), company_name=header.get('company_name', ''), ticker=header.get('ticker'), filing_date=header.get('filing_date', date.today()), period_end=header.get('period_end', date.today()), accession_number=header.get('accession_number', ''), word_count=len(text.split()))
        if filing_type in (_clI129c.FORM_10K, _clI129c.FORM_10Q):
            await self._parse_annual_quarterly(text, result)
        elif filing_type == _clI129c.FORM_8K:
            await self._parse_8k(text, result)
        elif filing_type == _clI129c.FORM_DEF14A:
            await self._parse_proxy(text, result)
        elif filing_type == _clI129c.FORM_13F:
            await self._parse_13f(text, result)
        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    def _fl102Af(self, _fIII2BO: str) -> str:
        _fIII2BO = re.sub('<[^>]+>', ' ', _fIII2BO)
        _fIII2BO = re.sub('\\s+', ' ', _fIII2BO)
        _fIII2BO = re.sub('&[a-zA-Z]+;', ' ', _fIII2BO)
        return _fIII2BO.strip()

    def _f1002Bl(self, _fIII2BO: str) -> _clI129c:
        text_upper = _fIII2BO[:5000].upper()
        if 'FORM 10-K' in text_upper or 'ANNUAL REPORT' in text_upper:
            return _clI129c.FORM_10K
        elif 'FORM 10-Q' in text_upper or 'QUARTERLY REPORT' in text_upper:
            return _clI129c.FORM_10Q
        elif 'FORM 8-K' in text_upper or 'CURRENT REPORT' in text_upper:
            return _clI129c.FORM_8K
        elif 'DEF 14A' in text_upper or 'PROXY STATEMENT' in text_upper:
            return _clI129c.FORM_DEF14A
        elif 'FORM 13F' in text_upper or '13F-HR' in text_upper:
            return _clI129c.FORM_13F
        elif 'FORM 4' in text_upper:
            return _clI129c.FORM_4
        elif 'FORM S-1' in text_upper or 'REGISTRATION STATEMENT' in text_upper:
            return _clI129c.FORM_S1
        return _clI129c.UNKNOWN

    def _fI1I2B2(self, _fIII2BO: str) -> Dict[str, Any]:
        header = {}
        cik_match = re.search('CENTRAL\\s+INDEX\\s+KEY:\\s*(\\d+)', _fIII2BO, re.IGNORECASE)
        if cik_match:
            header['cik'] = cik_match.group(1)
        name_match = re.search('COMPANY\\s+CONFORMED\\s+NAME:\\s*([^\\n]+)', _fIII2BO, re.IGNORECASE)
        if name_match:
            header['company_name'] = name_match.group(1).strip()
        date_match = re.search('FILED\\s+AS\\s+OF\\s+DATE:\\s*(\\d{8})', _fIII2BO, re.IGNORECASE)
        if date_match:
            d = date_match.group(1)
            header['filing_date'] = date(int(d[:4]), int(d[4:6]), int(d[6:8]))
        period_match = re.search('CONFORMED\\s+PERIOD\\s+OF\\s+REPORT:\\s*(\\d{8})', _fIII2BO, re.IGNORECASE)
        if period_match:
            d = period_match.group(1)
            header['period_end'] = date(int(d[:4]), int(d[4:6]), int(d[6:8]))
        acc_match = re.search('ACCESSION\\s+NUMBER:\\s*([\\d-]+)', _fIII2BO, re.IGNORECASE)
        if acc_match:
            header['accession_number'] = acc_match.group(1)
        ticker_match = re.search('TICKER\\s+SYMBOL:\\s*([A-Z]+)', _fIII2BO, re.IGNORECASE)
        if ticker_match:
            header['ticker'] = ticker_match.group(1)
        return header

    async def _f1012B3(self, _fIII2BO: str, _fl0l2B4: _cl002AB):
        for section_name, pattern in self._10k_sections.items():
            section_text = self._extract_section(_fIII2BO, pattern)
            if section_text:
                _fl0l2B4.sections[section_name] = section_text
        if 'risk_factors' in _fl0l2B4.sections:
            _fl0l2B4.risk_factors = self._parse_risk_factors(_fl0l2B4.sections['risk_factors'])
        if 'mda' in _fl0l2B4.sections:
            _fl0l2B4.mda_summary = self._summarize_mda(_fl0l2B4.sections['mda'])
            _fl0l2B4.mda_sentiment = self._analyze_mda_sentiment(_fl0l2B4.sections['mda'])
        if 'financials' in _fl0l2B4.sections:
            await self._parse_financial_statements(_fl0l2B4.sections['financials'], _fl0l2B4)
        _fl0l2B4.key_metrics = self._extract_key_metrics(_fl0l2B4)

    async def _fIOO2B5(self, _fIII2BO: str, _fl0l2B4: _cl002AB):
        for item in _c11029d:
            item_pattern = item.name.replace('_', '[\\s.]*')
            if re.search(item_pattern, _fIII2BO, re.IGNORECASE):
                _fl0l2B4.items_reported.append(item)
        for item in _fl0l2B4.items_reported:
            item_text = self._extract_8k_item(_fIII2BO, item)
            if item_text:
                _fl0l2B4.sections[item.name] = item_text

    async def _f1IO2B6(self, _fIII2BO: str, _fl0l2B4: _cl002AB):
        comp_section = self._extract_section(_fIII2BO, 'EXECUTIVE\\s+COMPENSATION')
        if comp_section:
            _fl0l2B4.executive_compensation = self._parse_compensation_table(comp_section)

    async def _fO102B7(self, _fIII2BO: str, _fl0l2B4: _cl002AB):
        pass

    def _flll2B8(self, _fIII2BO: str, _fOlI2B9: str) -> str:
        match = re.search(_fOlI2B9, _fIII2BO, re.IGNORECASE)
        if not match:
            return ''
        start = match.start()
        next_item = re.search('\\n\\s*ITEM\\s+\\d', _fIII2BO[start + 100:], re.IGNORECASE)
        if next_item:
            end = start + 100 + next_item.start()
        else:
            end = min(start + 50000, len(_fIII2BO))
        return _fIII2BO[start:end]

    def _fl0O2BA(self, _fIII2BO: str, _fllI2BB: _c11029d) -> str:
        item_num = _fllI2BB.name.replace('ITEM_', 'ITEM ')
        pattern = f'{item_num}[\\s.]*{_fllI2BB.value}'
        match = re.search(pattern, _fIII2BO, re.IGNORECASE)
        if not match:
            return ''
        start = match.start()
        next_item = re.search('ITEM\\s+\\d', _fIII2BO[start + 100:], re.IGNORECASE)
        end = start + 100 + next_item.start() if next_item else min(start + 10000, len(_fIII2BO))
        return _fIII2BO[start:end]

    def _fOlI2Bc(self, _fO0O2Bd: str) -> List[_cll02A8]:
        risks = []
        risk_splits = re.split('\\n\\s*([A-Z][^.]{20,100}\\.)\\s*\\n', _fO0O2Bd)
        for i in range(1, len(risk_splits), 2):
            if i + 1 < len(risk_splits):
                title = risk_splits[i].strip()
                description = risk_splits[i + 1][:2000].strip()
                category = self._categorize_risk(title + ' ' + description)
                severity = self._assess_risk_severity(description)
                risks.append(_cll02A8(title=title, description=description, category=category, severity=severity, keywords=self._extract_risk_keywords(description)))
        return risks[:50]

    def _fOlI2BE(self, _fIII2BO: str) -> str:
        text_lower = _fIII2BO.lower()
        if any((w in text_lower for w in ['market', 'economic', 'competition', 'industry'])):
            return 'Market'
        elif any((w in text_lower for w in ['operational', 'supply chain', 'manufacturing', 'technology'])):
            return 'Operational'
        elif any((w in text_lower for w in ['financial', 'debt', 'liquidity', 'credit'])):
            return 'Financial'
        elif any((w in text_lower for w in ['legal', 'regulatory', 'compliance', 'litigation'])):
            return 'Legal/Regulatory'
        elif any((w in text_lower for w in ['cyber', 'security', 'data', 'privacy'])):
            return 'Cybersecurity'
        elif any((w in text_lower for w in ['personnel', 'employee', 'talent', 'management'])):
            return 'Human Capital'
        else:
            return 'Other'

    def _flOI2Bf(self, _fIII2BO: str) -> str:
        text_lower = _fIII2BO.lower()
        high_severity = ['material adverse', 'significantly harm', 'substantial loss', 'could fail', 'bankruptcy', 'insolvency']
        medium_severity = ['may adversely', 'could harm', 'might affect', 'potential loss', 'uncertainty']
        if any((phrase in text_lower for phrase in high_severity)):
            return 'High'
        elif any((phrase in text_lower for phrase in medium_severity)):
            return 'Medium'
        else:
            return 'Low'

    def _fl102cO(self, _fIII2BO: str) -> List[str]:
        keywords = []
        important_terms = ['revenue', 'profit', 'loss', 'competition', 'regulation', 'technology', 'security', 'supply', 'demand', 'cost']
        text_lower = _fIII2BO.lower()
        for term in important_terms:
            if term in text_lower:
                keywords.append(term)
        return keywords

    def _fO1O2cl(self, _fO0O2Bd: str) -> str:
        paragraphs = _fO0O2Bd.split('\n\n')
        summary_parts = []
        total_len = 0
        for para in paragraphs:
            para = para.strip()
            if len(para) > 100 and total_len < 1000:
                summary_parts.append(para)
                total_len += len(para)
        return ' '.join(summary_parts)[:1500]

    def _fIIl2c2(self, _fO0O2Bd: str) -> Dict[str, float]:
        text_lower = _fO0O2Bd.lower()
        words = text_lower.split()
        total = len(words) if words else 1
        positive = ['growth', 'increase', 'improved', 'strong', 'exceeded', 'successful']
        negative = ['decline', 'decrease', 'challenging', 'difficult', 'loss', 'adverse']
        forward = ['expect', 'anticipate', 'believe', 'estimate', 'plan', 'intend']
        return {'positive': sum((1 for w in words if w in positive)) / total, 'negative': sum((1 for w in words if w in negative)) / total, 'forward_looking': sum((1 for w in words if w in forward)) / total}

    async def _f0Ol2c3(self, _fO0O2Bd: str, _fl0l2B4: _cl002AB):
        income = _c0OO29f(period_end=_fl0l2B4.period_end, is_quarterly=_fl0l2B4.filing_type == _clI129c.FORM_10Q)
        revenue_match = re.search('(?:total\\s+)?revenue[:\\s]+\\$?([\\d,]+)', _fO0O2Bd, re.IGNORECASE)
        if revenue_match:
            income.revenue = float(revenue_match.group(1).replace(',', ''))
        ni_match = re.search('net\\s+income[:\\s]+\\$?([\\d,]+)', _fO0O2Bd, re.IGNORECASE)
        if ni_match:
            income.net_income = float(ni_match.group(1).replace(',', ''))
        _fl0l2B4.income_statements.append(income)

    def _fll12c4(self, _fO0O2Bd: str) -> List[_cl1l2A9]:
        comps = []
        patterns = [('Chief Executive|CEO', 'CEO'), ('Chief Financial|CFO', 'CFO'), ('Chief Operating|COO', 'COO')]
        for pattern, title in patterns:
            match = re.search(pattern, _fO0O2Bd, re.IGNORECASE)
            if match:
                comps.append(_cl1l2A9(name=f'Executive ({title})', title=title, year=date.today().year))
        return comps

    def _fIOl2c5(self, _fl0l2B4: _cl002AB) -> Dict[str, Any]:
        metrics = {}
        if _fl0l2B4.income_statements:
            latest = _fl0l2B4.income_statements[-1]
            if latest.revenue:
                metrics['revenue'] = latest.revenue
            if latest.net_income:
                metrics['net_income'] = latest.net_income
            if latest._flIO2AO:
                metrics['gross_margin'] = latest._flIO2AO
            if latest._fOI02Al:
                metrics['operating_margin'] = latest._fOI02Al
        metrics['risk_count'] = len(_fl0l2B4.risk_factors)
        metrics['risk_high_count'] = sum((1 for r in _fl0l2B4.risk_factors if r.severity == 'High'))
        if _fl0l2B4.mda_sentiment:
            metrics['mda_sentiment'] = _fl0l2B4.mda_sentiment
        return metrics
__all__ = ['FilingType', 'ItemType8K', 'FinancialLineItem', 'IncomeStatement', 'BalanceSheet', 'CashFlowStatement', 'RiskFactor', 'ExecutiveCompensation', 'ParsedSECFiling', 'SECFilingProcessor']