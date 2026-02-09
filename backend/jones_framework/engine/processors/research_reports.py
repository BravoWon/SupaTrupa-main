from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum, auto
from datetime import datetime, timedelta
import re
import math
from collections import defaultdict
from jones_framework.core import bridge, ComponentRegistry

class _c1112c6(Enum):
    STRONG_BUY = 5
    BUY = 4
    OUTPERFORM = 4
    OVERWEIGHT = 4
    HOLD = 3
    NEUTRAL = 3
    MARKET_PERFORM = 3
    EQUAL_WEIGHT = 3
    UNDERPERFORM = 2
    UNDERWEIGHT = 2
    SELL = 1
    STRONG_SELL = 0

class _c1lI2c7(Enum):
    INITIATION = 'initiation'
    UPDATE = 'update'
    RATING_CHANGE = 'rating_change'
    TARGET_CHANGE = 'target_change'
    EARNINGS_PREVIEW = 'earnings_preview'
    EARNINGS_REVIEW = 'earnings_review'
    INDUSTRY_OVERVIEW = 'industry_overview'
    THEMATIC = 'thematic'
    QUANTITATIVE = 'quantitative'
    SPECIAL_SITUATIONS = 'special_situations'

class _c00O2c8(Enum):
    EARNINGS = 'earnings'
    PRODUCT_LAUNCH = 'product_launch'
    REGULATORY_APPROVAL = 'regulatory_approval'
    MERGER_ACQUISITION = 'merger_acquisition'
    MANAGEMENT_CHANGE = 'management_change'
    MARKET_SHARE_GAIN = 'market_share_gain'
    COST_REDUCTION = 'cost_reduction'
    CAPITAL_RETURN = 'capital_return'
    EXPANSION = 'expansion'
    RESTRUCTURING = 'restructuring'
    MACRO_TAILWIND = 'macro_tailwind'
    VALUATION_RERATING = 'valuation_rerating'

class _cIO12c9(Enum):
    EXECUTION = 'execution'
    COMPETITION = 'competition'
    REGULATORY = 'regulatory'
    MACRO = 'macro'
    CURRENCY = 'currency'
    COMMODITY = 'commodity'
    TECHNOLOGY = 'technology'
    MANAGEMENT = 'management'
    LIQUIDITY = 'liquidity'
    VALUATION = 'valuation'
    ESG = 'esg'
    GEOPOLITICAL = 'geopolitical'

@dataclass
class _cII02cA:
    name: str
    firm: str
    email: Optional[str] = None
    sectors: List[str] = field(default_factory=list)
    years_experience: Optional[int] = None
    accuracy_score: float = 0.5
    target_accuracy: float = 0.5
    timeliness_score: float = 0.5
    consistency_score: float = 0.5

    @property
    def _f1012cB(self) -> float:
        return self.accuracy_score * 0.4 + self.target_accuracy * 0.3 + self.timeliness_score * 0.2 + self.consistency_score * 0.1

@dataclass
class _cOOI2cc:
    name: str
    value: float
    unit: str
    confidence: float = 0.5
    sensitivity: float = 0.5
    historical_avg: Optional[float] = None
    industry_avg: Optional[float] = None
    bear_case: Optional[float] = None
    bull_case: Optional[float] = None

@dataclass
class _c0l02cd:
    metric: str
    period: str
    estimate: float
    unit: str
    prior_estimate: Optional[float] = None
    consensus: Optional[float] = None
    vs_consensus: Optional[float] = None
    year_ago: Optional[float] = None
    yoy_growth: Optional[float] = None

@dataclass
class _c10O2cE:
    target: float
    current_price: float
    upside_pct: float
    methodology: str
    timeframe: str
    bear_case: Optional[float] = None
    base_case: float = 0.0
    bull_case: Optional[float] = None
    prior_target: Optional[float] = None
    change_pct: Optional[float] = None

@dataclass
class _cOOO2cf:
    summary: str
    key_points: List[str]
    catalysts: List[Tuple[_c00O2c8, str, Optional[datetime]]]
    risks: List[Tuple[_cIO12c9, str, float]]
    time_horizon: str
    conviction_level: float

@dataclass
class _cO112dO:
    name: str
    ticker: str
    market_cap: Optional[float] = None
    ev_ebitda: Optional[float] = None
    pe_ratio: Optional[float] = None
    price_sales: Optional[float] = None
    revenue_growth: Optional[float] = None
    margin: Optional[float] = None
    rating: Optional[_c1112c6] = None

@dataclass
class _cOll2dl:
    report_id: str
    title: str
    report_type: _c1lI2c7
    company: str
    ticker: str
    sector: str
    industry: str
    analyst: _cII02cA
    published_at: datetime
    pages: int = 0
    rating: _c1112c6
    prior_rating: Optional[_c1112c6] = None
    rating_changed: bool = False
    price_target: _c10O2cE
    thesis: _cOOO2cf
    estimates: List[_c0l02cd] = field(default_factory=list)
    assumptions: List[_cOOI2cc] = field(default_factory=list)
    peer_group: List[_cO112dO] = field(default_factory=list)
    valuation_premium: Optional[float] = None
    depth_score: float = 0.5
    data_quality_score: float = 0.5
    objectivity_score: float = 0.5
    key_quotes: List[str] = field(default_factory=list)
    full_text: str = ''

@dataclass
class _cOOI2d2:
    ticker: str
    as_of: datetime
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    mean_rating: float = 0.0
    median_target: float = 0.0
    mean_target: float = 0.0
    high_target: float = 0.0
    low_target: float = 0.0
    target_spread: float = 0.0
    upgrades_30d: int = 0
    downgrades_30d: int = 0
    target_increases_30d: int = 0
    target_decreases_30d: int = 0

    @property
    def _fOIl2d3(self) -> int:
        return self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell

    @property
    def _flI02d4(self) -> float:
        if self._fOIl2d3 == 0:
            return 0
        return (self.strong_buy + self.buy) / self._fOIl2d3

    @property
    def _fOI02d5(self) -> int:
        return self.upgrades_30d - self.downgrades_30d + self.target_increases_30d - self.target_decreases_30d

class _clO12d6:
    RATING_PATTERNS = {_c1112c6.STRONG_BUY: ['\\bstrong\\s+buy\\b', '\\btop\\s+pick\\b', '\\bconviction\\s+buy\\b'], _c1112c6.BUY: ['\\bbuy\\b(?!\\s+hold)', '\\boverweight\\b', '\\boutperform\\b', '\\baccumulate\\b'], _c1112c6.HOLD: ['\\bhold\\b', '\\bneutral\\b', '\\bmarket\\s+perform\\b', '\\bequal[\\s-]weight\\b'], _c1112c6.UNDERPERFORM: ['\\bunderperform\\b', '\\bunderweight\\b', '\\breduce\\b'], _c1112c6.SELL: ['\\bsell\\b(?!\\s+hold)', '\\bstrong\\s+sell\\b']}

    def __init__(self):
        self._patterns: Dict[_c1112c6, List[re.Pattern]] = {}
        for rating, patterns in self.RATING_PATTERNS.items():
            self._patterns[rating] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def process_batch(self, _flOO2d8: str) -> Optional[_c1112c6]:
        text_lower = _flOO2d8.lower()
        for rating in [_c1112c6.STRONG_BUY, _c1112c6.STRONG_SELL, _c1112c6.BUY, _c1112c6.SELL, _c1112c6.HOLD, _c1112c6.UNDERPERFORM]:
            if rating in self._patterns:
                for pattern in self._patterns[rating]:
                    if pattern.search(text_lower):
                        return rating
        return None

class _cI1l2d9:
    TARGET_PATTERNS = [re.compile('(?:price\\s+)?target\\s*(?:of|:)?\\s*\\$?(\\d+(?:\\.\\d+)?)', re.IGNORECASE), re.compile('\\$(\\d+(?:\\.\\d+)?)\\s*(?:price\\s+)?target', re.IGNORECASE), re.compile('(?:pt|target):\\s*\\$?(\\d+(?:\\.\\d+)?)', re.IGNORECASE)]
    METHODOLOGY_PATTERNS = {'DCF': ['\\bdcf\\b', '\\bdiscounted\\s+cash\\s+flow\\b'], 'Sum-of-Parts': ['\\bsotp\\b', '\\bsum[\\s-]of[\\s-](?:the[\\s-])?parts\\b'], 'Comparable': ['\\bcomparable\\b', '\\bpeer\\s+multiple\\b', '\\btrading\\s+multiple\\b'], 'DDM': ['\\bddm\\b', '\\bdividend\\s+discount\\b'], 'NAV': ['\\bnav\\b', '\\bnet\\s+asset\\s+value\\b'], 'P/E Multiple': ['\\bp/?e\\s+(?:of\\s+)?(\\d+)', '\\bearnings\\s+multiple\\b'], 'EV/EBITDA': ['\\bev/?ebitda\\s+(?:of\\s+)?(\\d+)', '\\bebitda\\s+multiple\\b']}

    def __init__(self):
        self._methodology_patterns: Dict[str, List[re.Pattern]] = {}
        for method, patterns in self.METHODOLOGY_PATTERNS.items():
            self._methodology_patterns[method] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def _fI012dA(self, _flOO2d8: str, _fI0l2dB: float) -> Optional[_c10O2cE]:
        target_value = None
        for pattern in self.TARGET_PATTERNS:
            match = pattern.search(_flOO2d8)
            if match:
                target_value = float(match.group(1))
                break
        if target_value is None:
            return None
        methodology = 'Not specified'
        for method, patterns in self._methodology_patterns.items():
            for pattern in patterns:
                if pattern.search(_flOO2d8):
                    methodology = method
                    break
            if methodology != 'Not specified':
                break
        upside = (target_value - _fI0l2dB) / _fI0l2dB * 100 if _fI0l2dB else 0
        return _c10O2cE(target=target_value, current_price=_fI0l2dB, upside_pct=upside, methodology=methodology, timeframe='12-month', base_case=target_value)

class _cllO2dc:
    CATALYST_PATTERNS: Dict[_c00O2c8, List[str]] = {_c00O2c8.EARNINGS: ['earnings\\s+(?:beat|growth|surprise|momentum)'], _c00O2c8.PRODUCT_LAUNCH: ['(?:new|upcoming)\\s+product', 'product\\s+launch', 'release\\s+of'], _c00O2c8.REGULATORY_APPROVAL: ['(?:fda|regulatory)\\s+approval', 'clearance'], _c00O2c8.MERGER_ACQUISITION: ['(?:m&a|merger|acquisition)', 'takeover\\s+target'], _c00O2c8.MARKET_SHARE_GAIN: ['market\\s+share\\s+(?:gain|growth)', 'taking\\s+share'], _c00O2c8.COST_REDUCTION: ['cost\\s+(?:cut|reduction|savings)', 'margin\\s+expansion'], _c00O2c8.CAPITAL_RETURN: ['(?:dividend|buyback)', 'capital\\s+return'], _c00O2c8.VALUATION_RERATING: ['(?:multiple|valuation)\\s+expansion', 'rerating']}
    RISK_PATTERNS: Dict[_cIO12c9, List[str]] = {_cIO12c9.EXECUTION: ['execution\\s+risk', 'implementation\\s+challenges'], _cIO12c9.COMPETITION: ['competitive\\s+(?:pressure|threat)', 'market\\s+share\\s+loss'], _cIO12c9.REGULATORY: ['regulatory\\s+(?:risk|overhang)', 'government\\s+action'], _cIO12c9.MACRO: ['(?:macro|economic)\\s+(?:downturn|weakness)', 'recession'], _cIO12c9.CURRENCY: ['(?:currency|fx)\\s+(?:risk|exposure)', 'dollar\\s+strength'], _cIO12c9.TECHNOLOGY: ['(?:tech|technological)\\s+(?:disruption|obsolescence)'], _cIO12c9.MANAGEMENT: ['(?:management|key\\s+person)\\s+(?:risk|departure)'], _cIO12c9.VALUATION: ['(?:valuation|multiple)\\s+(?:risk|compression)', 'overvalued']}

    def __init__(self):
        self._catalyst_patterns: Dict[_c00O2c8, List[re.Pattern]] = {}
        self._risk_patterns: Dict[_cIO12c9, List[re.Pattern]] = {}
        for cat, patterns in self.CATALYST_PATTERNS.items():
            self._catalyst_patterns[cat] = [re.compile(p, re.IGNORECASE) for p in patterns]
        for risk, patterns in self.RISK_PATTERNS.items():
            self._risk_patterns[risk] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def _fI012dA(self, _flOO2d8: str) -> _cOOO2cf:
        catalysts = []
        for cat_type, patterns in self._catalyst_patterns.items():
            for pattern in patterns:
                match = pattern.search(_flOO2d8)
                if match:
                    start = max(0, match.start() - 50)
                    end = min(len(_flOO2d8), match.end() + 100)
                    context = _flOO2d8[start:end].strip()
                    catalysts.append((cat_type, context, None))
                    break
        risks = []
        for risk_type, patterns in self._risk_patterns.items():
            for pattern in patterns:
                match = pattern.search(_flOO2d8)
                if match:
                    start = max(0, match.start() - 50)
                    end = min(len(_flOO2d8), match.end() + 100)
                    context = _flOO2d8[start:end].strip()
                    severity = 0.5
                    risks.append((risk_type, context, severity))
                    break
        key_points = []
        bullet_pattern = re.compile('(?:^|\\n)\\s*[•\\-\\*]\\s*(.+?)(?:\\n|$)')
        for match in bullet_pattern.finditer(_flOO2d8)[:10]:
            point = match.group(1).strip()
            if len(point) > 20:
                key_points.append(point)
        paragraphs = _flOO2d8.split('\n\n')
        summary = ''
        for para in paragraphs:
            if len(para) > 100:
                summary = para[:500].strip()
                break
        return _cOOO2cf(summary=summary, key_points=key_points[:5], catalysts=catalysts, risks=risks, time_horizon=self._extract_time_horizon(_flOO2d8), conviction_level=self._estimate_conviction(_flOO2d8))

    def _f1l12dd(self, _flOO2d8: str) -> str:
        text_lower = _flOO2d8.lower()
        if any((term in text_lower for term in ['near-term', 'short-term', '3-6 month'])):
            return 'Short-term'
        elif any((term in text_lower for term in ['long-term', 'multi-year', '3-5 year'])):
            return 'Long-term'
        else:
            return 'Medium-term'

    def _fI102dE(self, _flOO2d8: str) -> float:
        text_lower = _flOO2d8.lower()
        high_conviction_words = ['conviction', 'confident', 'strongly', 'clearly', 'top pick', 'best idea', 'compelling']
        low_conviction_words = ['uncertain', 'cautious', 'concerned', 'risk', 'wait and see', 'limited visibility']
        high_count = sum((1 for word in high_conviction_words if word in text_lower))
        low_count = sum((1 for word in low_conviction_words if word in text_lower))
        base = 0.5
        score = base + high_count * 0.1 - low_count * 0.1
        return max(0.1, min(0.95, score))

class _cOl02df:
    METRIC_PATTERNS = {'Revenue': ['revenue\\s+(?:of\\s+)?\\$?(\\d+(?:\\.\\d+)?)\\s*(million|billion|M|B)?'], 'EPS': ['eps\\s+(?:of\\s+)?\\$?(\\d+\\.\\d+)'], 'EBITDA': ['ebitda\\s+(?:of\\s+)?\\$?(\\d+(?:\\.\\d+)?)\\s*(million|billion|M|B)?'], 'Net Income': ['net\\s+income\\s+(?:of\\s+)?\\$?(\\d+(?:\\.\\d+)?)\\s*(million|billion|M|B)?'], 'FCF': ['(?:free\\s+cash\\s+flow|fcf)\\s+(?:of\\s+)?\\$?(\\d+(?:\\.\\d+)?)\\s*(million|billion|M|B)?'], 'Gross Margin': ['gross\\s+margin\\s+(?:of\\s+)?(\\d+(?:\\.\\d+)?)\\s*%'], 'Operating Margin': ['operating\\s+margin\\s+(?:of\\s+)?(\\d+(?:\\.\\d+)?)\\s*%']}
    PERIOD_PATTERNS = ["(Q[1-4])\\s*\\'?(\\d{2,4})", "(FY|CY)\\s*\\'?(\\d{2,4})", '(\\d{4})\\s*(Q[1-4])?']

    def __init__(self):
        self._metric_patterns: Dict[str, List[re.Pattern]] = {}
        for metric, patterns in self.METRIC_PATTERNS.items():
            self._metric_patterns[metric] = [re.compile(p, re.IGNORECASE) for p in patterns]
        self._period_patterns = [re.compile(p, re.IGNORECASE) for p in self.PERIOD_PATTERNS]

    def _fI012dA(self, _flOO2d8: str) -> List[_c0l02cd]:
        estimates = []
        for metric, patterns in self._metric_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(_flOO2d8):
                    value = float(match.group(1))
                    unit = match.group(2) if len(match.groups()) > 1 else ''
                    if unit and unit.lower() in ['billion', 'b']:
                        value *= 1000000000.0
                        unit = 'USD'
                    elif unit and unit.lower() in ['million', 'm']:
                        value *= 1000000.0
                        unit = 'USD'
                    elif '%' in metric.lower() or 'margin' in metric.lower():
                        unit = '%'
                    else:
                        unit = 'USD'
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(_flOO2d8), match.end() + 50)
                    context = _flOO2d8[context_start:context_end]
                    period = self._extract_period(context)
                    estimates.append(_c0l02cd(metric=metric, period=period, estimate=value, unit=unit))
        return estimates

    def _fIlI2EO(self, _flOO2d8: str) -> str:
        for pattern in self._period_patterns:
            match = pattern.search(_flOO2d8)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return f"{groups[0]} {groups[1] or ''}".strip()
                return groups[0]
        return 'FY Current'

class _cl0I2El:

    def __init__(self):
        self._rating_parser = _clO12d6()
        self._target_extractor = _cI1l2d9()
        self._thesis_extractor = _cllO2dc()
        self._estimate_extractor = _cOl02df()
        self._registry = ComponentRegistry.get_instance()

    @bridge(connects_to=['ConditionState', 'ActivityState', 'LinguisticArbitrage', 'CorrelationCutter', 'DocumentProcessor', 'NewsArticleProcessor'], connection_types={'ConditionState': 'emits', 'ActivityState': 'triggers', 'LinguisticArbitrage': 'feeds', 'CorrelationCutter': 'signals', 'DocumentProcessor': 'extends', 'NewsArticleProcessor': 'complements'})
    def _f10I2E2(self, _flOO2d8: str, _fll02E3: str, _f10O2E4: str, _fO002E5: str, _fO0I2E6: str, _f1002E7: str, _fIII2E8: datetime, _fI0l2dB: float, _fl112E9: str='', _f1Il2EA: str='') -> _cOll2dl:
        import hashlib
        report_id = hashlib.md5(f'{_fO002E5}{_f1002E7}{_fIII2E8}'.encode()).hexdigest()[:16]
        rating = self._rating_parser.process_batch(_flOO2d8) or _c1112c6.HOLD
        rating_changed = False
        prior_rating = None
        if 'upgrade' in _flOO2d8.lower() or 'downgrade' in _flOO2d8.lower():
            rating_changed = True
        price_target = self._target_extractor._fI012dA(_flOO2d8, _fI0l2dB)
        if price_target is None:
            price_target = _c10O2cE(target=_fI0l2dB, current_price=_fI0l2dB, upside_pct=0, methodology='Not specified', timeframe='12-month')
        thesis = self._thesis_extractor._fI012dA(_flOO2d8)
        estimates = self._estimate_extractor._fI012dA(_flOO2d8)
        report_type = self._classify_report_type(_flOO2d8, rating_changed)
        analyst = _cII02cA(name=_fO0I2E6, firm=_f1002E7)
        depth_score = self._calculate_depth_score(_flOO2d8, estimates, thesis)
        objectivity_score = self._calculate_objectivity_score(_flOO2d8, thesis)
        key_quotes = self._extract_key_quotes(_flOO2d8)
        return _cOll2dl(report_id=report_id, title=_fll02E3, report_type=report_type, company=_f10O2E4, ticker=_fO002E5, sector=_fl112E9, industry=_f1Il2EA, analyst=analyst, published_at=_fIII2E8, rating=rating, prior_rating=prior_rating, rating_changed=rating_changed, price_target=price_target, thesis=thesis, estimates=estimates, depth_score=depth_score, objectivity_score=objectivity_score, key_quotes=key_quotes, full_text=_flOO2d8)

    def _fI1O2EB(self, _flOO2d8: str, _fO112Ec: bool) -> _c1lI2c7:
        text_lower = _flOO2d8.lower()
        if any((term in text_lower for term in ['initiat', 'coverage'])):
            return _c1lI2c7.INITIATION
        elif _fO112Ec:
            return _c1lI2c7.RATING_CHANGE
        elif 'target' in text_lower and ('raise' in text_lower or 'lower' in text_lower):
            return _c1lI2c7.TARGET_CHANGE
        elif any((term in text_lower for term in ['earnings preview', 'ahead of earnings'])):
            return _c1lI2c7.EARNINGS_PREVIEW
        elif any((term in text_lower for term in ['earnings review', 'post-earnings', 'results'])):
            return _c1lI2c7.EARNINGS_REVIEW
        elif any((term in text_lower for term in ['industry', 'sector', 'overview'])):
            return _c1lI2c7.INDUSTRY_OVERVIEW
        else:
            return _c1lI2c7.UPDATE

    def _fI1l2Ed(self, _flOO2d8: str, _f0Il2EE: List[_c0l02cd], _flII2Ef: _cOOO2cf) -> float:
        score = 0.3
        word_count = len(_flOO2d8.split())
        score += min(0.2, word_count / 5000)
        score += min(0.15, len(_f0Il2EE) * 0.02)
        if _flII2Ef.catalysts:
            score += 0.1
        if _flII2Ef.risks:
            score += 0.1
        if _flII2Ef.key_points:
            score += 0.05
        data_terms = ['exhibit', 'figure', 'chart', 'table', 'model']
        data_count = sum((1 for term in data_terms if term in _flOO2d8.lower()))
        score += min(0.1, data_count * 0.02)
        return min(1.0, score)

    def _flll2fO(self, _flOO2d8: str, _flII2Ef: _cOOO2cf) -> float:
        score = 0.5
        if _flII2Ef.catalysts and _flII2Ef.risks:
            ratio = len(_flII2Ef.risks) / len(_flII2Ef.catalysts)
            if 0.5 <= ratio <= 2.0:
                score += 0.2
            else:
                score -= 0.1
        hedging_terms = ['however', 'although', 'risk', 'concern', 'challenge']
        hedging_count = sum((1 for term in hedging_terms if term in _flOO2d8.lower()))
        score += min(0.15, hedging_count * 0.03)
        extreme_terms = ['definitely', 'certainly', 'absolutely', 'guaranteed']
        extreme_count = sum((1 for term in extreme_terms if term in _flOO2d8.lower()))
        score -= extreme_count * 0.05
        return max(0.1, min(0.95, score))

    def _f1OO2fl(self, _flOO2d8: str) -> List[str]:
        quotes = []
        quote_pattern = re.compile('"([^"]{30,200})"')
        for match in quote_pattern.finditer(_flOO2d8)[:5]:
            quotes.append(match.group(1))
        belief_pattern = re.compile('(?:we believe|our view is that|we expect|we think)\\s+([^.]{20,150})\\.', re.IGNORECASE)
        for match in belief_pattern.finditer(_flOO2d8)[:3]:
            quotes.append(match.group(1))
        return quotes[:5]

class _c1lO2f2:

    def __init__(self):
        self._reports: Dict[str, List[_cOll2dl]] = defaultdict(list)
        self._latest_by_analyst: Dict[str, Dict[str, _cOll2dl]] = defaultdict(dict)

    def _fl0l2f3(self, _f0O12f4: _cOll2dl):
        self._reports[_f0O12f4._fO002E5].append(_f0O12f4)
        key = f'{_f0O12f4.analyst.firm}_{_f0O12f4.analyst.name}'
        self._latest_by_analyst[_f0O12f4._fO002E5][key] = _f0O12f4

    def _f01l2f5(self, _fO002E5: str) -> _cOOI2d2:
        reports = list(self._latest_by_analyst.get(_fO002E5, {}).values())
        if not reports:
            return _cOOI2d2(ticker=_fO002E5, as_of=datetime.now())
        rating_counts = defaultdict(int)
        for _f0O12f4 in reports:
            if _f0O12f4.rating.value >= 4:
                if _f0O12f4.rating == _c1112c6.STRONG_BUY:
                    rating_counts['strong_buy'] += 1
                else:
                    rating_counts['buy'] += 1
            elif _f0O12f4.rating.value == 3:
                rating_counts['hold'] += 1
            elif _f0O12f4.rating.value == 2:
                rating_counts['sell'] += 1
            else:
                rating_counts['strong_sell'] += 1
        targets = [r.price_target.target for r in reports if r.price_target.target > 0]
        return _cOOI2d2(ticker=_fO002E5, as_of=datetime.now(), strong_buy=rating_counts['strong_buy'], buy=rating_counts['buy'], hold=rating_counts['hold'], sell=rating_counts['sell'], strong_sell=rating_counts['strong_sell'], mean_rating=sum((r.rating.value for r in reports)) / len(reports), median_target=sorted(targets)[len(targets) // 2] if targets else 0, mean_target=sum(targets) / len(targets) if targets else 0, high_target=max(targets) if targets else 0, low_target=min(targets) if targets else 0, target_spread=(max(targets) - min(targets)) / min(targets) if targets and min(targets) > 0 else 0)

    def _fl0I2f6(self, _fO002E5: str, _f1I12f7: int=30) -> Dict[str, Any]:
        cutoff = datetime.now() - timedelta(days=_f1I12f7)
        reports = [r for r in self._reports.get(_fO002E5, []) if r._fIII2E8 > cutoff]
        upgrades = sum((1 for r in reports if r._fO112Ec and r.rating.value >= 4))
        downgrades = sum((1 for r in reports if r._fO112Ec and r.rating.value <= 2))
        target_changes = []
        for r in reports:
            if r.price_target.change_pct is not None:
                target_changes.append(r.price_target.change_pct)
        return {'ticker': _fO002E5, 'period_days': _f1I12f7, 'reports_count': len(reports), 'upgrades': upgrades, 'downgrades': downgrades, 'net_revisions': upgrades - downgrades, 'avg_target_change': sum(target_changes) / len(target_changes) if target_changes else 0}

class _c0l02f8:

    def __init__(self):
        self._analyst_history: Dict[str, List[Dict]] = defaultdict(list)

    def _f0II2f9(self, _f0ll2fA: _cII02cA, _fO002E5: str, _f0OO2fB: _c1112c6, _fO1O2fc: float, _f0O02fd: float, _f1012fE: datetime):
        key = f'{_f0ll2fA.firm}_{_f0ll2fA.name}'
        self._analyst_history[key].append({'ticker': _fO002E5, 'rating': _f0OO2fB, 'target': _fO1O2fc, 'price_at_rec': _f0O02fd, 'date': _f1012fE, 'outcome_recorded': False})

    def _fl102ff(self, _f0ll2fA: _cII02cA, _fO002E5: str, _f1012fE: datetime, _fOII3OO: float):
        key = f'{_f0ll2fA.firm}_{_f0ll2fA.name}'
        history = self._analyst_history.get(key, [])
        for rec in history:
            if rec['ticker'] == _fO002E5 and (not rec['outcome_recorded']) and (rec['date'] < _f1012fE):
                rec['actual_price'] = _fOII3OO
                rec['target_accuracy'] = 1 - abs(rec['target'] - _fOII3OO) / rec['price_at_rec']
                rec['direction_correct'] = rec['rating'].value >= 4 and _fOII3OO > rec['price_at_rec'] or (rec['rating'].value <= 2 and _fOII3OO < rec['price_at_rec']) or rec['rating'].value == 3
                rec['outcome_recorded'] = True

    def _flOO3Ol(self, _f0ll2fA: _cII02cA) -> Dict[str, float]:
        key = f'{_f0ll2fA.firm}_{_f0ll2fA.name}'
        history = [r for r in self._analyst_history.get(key, []) if r.get('outcome_recorded')]
        if not history:
            return {'accuracy': 0.5, 'target_accuracy': 0.5, 'total_recommendations': 0}
        direction_correct = sum((1 for r in history if r['direction_correct']))
        avg_target_accuracy = sum((r['target_accuracy'] for r in history)) / len(history)
        return {'accuracy': direction_correct / len(history), 'target_accuracy': max(0, min(1, avg_target_accuracy)), 'total_recommendations': len(history)}

def _f00I3O2() -> _cl0I2El:
    return _cl0I2El()

def _fOOO3O3() -> _c1lO2f2:
    return _c1lO2f2()

def _flOO3O4() -> _c0l02f8:
    return _c0l02f8()