from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum, auto
from datetime import datetime, timedelta
import re
import math
from collections import defaultdict
import hashlib
from jones_framework.core import bridge, ComponentRegistry

class _c00l343(Enum):
    WIRE_SERVICE = 'wire_service'
    FINANCIAL_NEWS = 'financial_news'
    MAINSTREAM_MEDIA = 'mainstream_media'
    TRADE_PUBLICATION = 'trade_publication'
    REGULATORY = 'regulatory'
    COMPANY_PR = 'company_pr'
    SOCIAL_MEDIA = 'social_media'
    BLOG = 'blog'
    AGGREGATOR = 'aggregator'
    UNKNOWN = 'unknown'

class _cIIO344(Enum):
    EARNINGS = 'earnings'
    MERGER_ACQUISITION = 'merger_acquisition'
    PRODUCT_LAUNCH = 'product_launch'
    EXECUTIVE_CHANGE = 'executive_change'
    REGULATORY_ACTION = 'regulatory_action'
    LEGAL_ACTION = 'legal_action'
    PARTNERSHIP = 'partnership'
    EXPANSION = 'expansion'
    RESTRUCTURING = 'restructuring'
    DIVIDEND = 'dividend'
    STOCK_BUYBACK = 'stock_buyback'
    IPO_OFFERING = 'ipo_offering'
    BANKRUPTCY = 'bankruptcy'
    DATA_BREACH = 'data_breach'
    LABOR_DISPUTE = 'labor_dispute'
    ENVIRONMENTAL = 'environmental'
    GEOPOLITICAL = 'geopolitical'
    MACRO_ECONOMIC = 'macro_economic'
    ANALYST_RATING = 'analyst_rating'
    INSIDER_TRADING = 'insider_trading'
    GENERAL = 'general'

class _cI1I345(Enum):
    BREAKING = 5
    URGENT = 4
    DEVELOPING = 3
    STANDARD = 2
    BACKGROUND = 1

class _c0Ol346(Enum):
    VERY_POSITIVE = 2
    POSITIVE = 1
    NEUTRAL = 0
    NEGATIVE = -1
    VERY_NEGATIVE = -2

@dataclass
class _clII347:
    text: str
    entity_type: str
    normalized_name: Optional[str] = None
    ticker: Optional[str] = None
    confidence: float = 1.0
    mentions: int = 1
    sentiment_association: float = 0.0

@dataclass
class _c1I0348:
    event_type: _cIIO344
    description: str
    entities_involved: List[_clII347]
    timestamp: Optional[datetime] = None
    confidence: float = 1.0
    magnitude: float = 0.5
    direction: Optional[str] = None

@dataclass
class _cIlO349:
    name: str
    source_type: _c00l343
    accuracy_score: float = 0.8
    speed_score: float = 0.5
    bias_score: float = 0.5
    influence_score: float = 0.5
    overall_score: float = 0.6

@dataclass
class _cI0O34A:
    article_id: str
    url: Optional[str] = None
    title: str = ''
    summary: str = ''
    body: str = ''
    source: str = ''
    source_type: _c00l343 = _c00l343.UNKNOWN
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    fetched_at: datetime = field(default_factory=datetime.now)
    entities: List[_clII347] = field(default_factory=list)
    events: List[_c1I0348] = field(default_factory=list)
    tickers_mentioned: List[str] = field(default_factory=list)
    sentiment: _c0Ol346 = _c0Ol346.NEUTRAL
    sentiment_score: float = 0.0
    urgency: _cI1I345 = _cI1I345.STANDARD
    relevance_score: float = 0.5
    credibility_score: float = 0.5
    related_articles: List[str] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)
    confirms: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.article_id:
            content = f'{self.title}{self.source}{self.published_at}'
            self.article_id = hashlib.md5(content.encode()).hexdigest()[:16]

@dataclass
class _cIOO34B:
    cluster_id: str
    topic: str
    articles: List[_cI0O34A]
    primary_entities: List[_clII347]
    primary_event: Optional[_c1I0348] = None
    aggregate_sentiment: float = 0.0
    source_diversity: float = 0.0
    first_reported: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    is_developing: bool = False

@dataclass
class _cOOO34c:
    signal_id: str
    signal_type: str
    ticker: str
    direction: str
    strength: float
    confidence: float
    urgency: _cI1I345
    source_articles: List[str]
    generated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    rationale: str = ''

class _cIII34d:
    _KNOWN_SOURCES: Dict[str, _cIlO349] = {'reuters': _cIlO349(name='Reuters', source_type=_c00l343.WIRE_SERVICE, accuracy_score=0.95, speed_score=0.95, bias_score=0.85, influence_score=0.9, overall_score=0.91), 'bloomberg': _cIlO349(name='Bloomberg', source_type=_c00l343.FINANCIAL_NEWS, accuracy_score=0.92, speed_score=0.9, bias_score=0.8, influence_score=0.95, overall_score=0.89), 'wsj': _cIlO349(name='Wall Street Journal', source_type=_c00l343.FINANCIAL_NEWS, accuracy_score=0.9, speed_score=0.75, bias_score=0.7, influence_score=0.85, overall_score=0.8), 'ft': _cIlO349(name='Financial Times', source_type=_c00l343.FINANCIAL_NEWS, accuracy_score=0.9, speed_score=0.7, bias_score=0.75, influence_score=0.8, overall_score=0.79), 'cnbc': _cIlO349(name='CNBC', source_type=_c00l343.FINANCIAL_NEWS, accuracy_score=0.8, speed_score=0.85, bias_score=0.65, influence_score=0.75, overall_score=0.76), 'sec': _cIlO349(name='SEC', source_type=_c00l343.REGULATORY, accuracy_score=1.0, speed_score=0.3, bias_score=1.0, influence_score=0.95, overall_score=0.81), 'twitter': _cIlO349(name='Twitter/X', source_type=_c00l343.SOCIAL_MEDIA, accuracy_score=0.4, speed_score=1.0, bias_score=0.3, influence_score=0.6, overall_score=0.58), 'reddit': _cIlO349(name='Reddit', source_type=_c00l343.SOCIAL_MEDIA, accuracy_score=0.35, speed_score=0.9, bias_score=0.25, influence_score=0.45, overall_score=0.49)}

    @classmethod
    def _fOO134E(cls, _f10O34f: str) -> _cIlO349:
        normalized = _f10O34f.lower().replace(' ', '').replace('.', '')
        for key, cred in cls._KNOWN_SOURCES.items():
            if key in normalized:
                return cred
        if any((wire in normalized for wire in ['ap', 'afp', 'efe'])):
            return _cIlO349(name=_f10O34f, source_type=_c00l343.WIRE_SERVICE, overall_score=0.85)
        elif any((pr in normalized for pr in ['prnewswire', 'businesswire', 'globenewswire'])):
            return _cIlO349(name=_f10O34f, source_type=_c00l343.COMPANY_PR, overall_score=0.6)
        return _cIlO349(name=_f10O34f, source_type=_c00l343.UNKNOWN, overall_score=0.5)

class _c0I135O:
    TICKER_PATTERN = re.compile('\\b([A-Z]{1,5})\\b(?:\\s*\\((?:NYSE|NASDAQ|AMEX|NYSE\\s*Arca)\\))?')
    COMPANY_SUFFIXES = ['Inc', 'Corp', 'Corporation', 'Ltd', 'Limited', 'LLC', 'LP', 'PLC', 'AG', 'SA', 'NV', 'SE', 'Co', 'Company', 'Group', 'Holdings', 'Holding', 'Technologies', 'Technology', 'Systems']
    PERSON_TITLES = ['CEO', 'CFO', 'COO', 'CTO', 'CMO', 'CIO', 'CISO', 'President', 'Chairman', 'Director', 'VP', 'Vice President', 'Chief', 'Head of', 'Managing Director', 'Partner', 'Analyst', 'Strategist', 'Economist', 'Secretary']
    _TICKER_MAP = {'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corporation', 'GOOGL': 'Alphabet Inc.', 'AMZN': 'Amazon.com Inc.', 'META': 'Meta Platforms Inc.', 'TSLA': 'Tesla Inc.', 'NVDA': 'NVIDIA Corporation', 'JPM': 'JPMorgan Chase & Co.', 'V': 'Visa Inc.', 'JNJ': 'Johnson & Johnson', 'WMT': 'Walmart Inc.', 'XOM': 'Exxon Mobil Corporation'}

    def __init__(self):
        self._company_pattern = re.compile(f"\\b([A-Z][a-zA-Z]+(?:\\s+[A-Z][a-zA-Z]+)*)\\s+({'|'.join(self.COMPANY_SUFFIXES)})" + '\\.?\\b')
        self._person_pattern = re.compile(f"\\b([A-Z][a-z]+\\s+[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)?)\\s*,?\\s*(?:{'|'.join(self.PERSON_TITLES)})" + '\\b')

    def process_batch(self, _f1OO352: str) -> List[_clII347]:
        entities = []
        for match in self._company_pattern.finditer(_f1OO352):
            name = f'{match.group(1)} {match.group(2)}'
            entities.append(_clII347(text=name, entity_type='COMPANY', normalized_name=name.strip(), confidence=0.9))
        for match in self._person_pattern.finditer(_f1OO352):
            entities.append(_clII347(text=match.group(1), entity_type='PERSON', confidence=0.85))
        for match in self.TICKER_PATTERN.finditer(_f1OO352):
            ticker = match.group(1)
            if ticker in self._TICKER_MAP:
                entities.append(_clII347(text=ticker, entity_type='TICKER', ticker=ticker, normalized_name=self._TICKER_MAP[ticker], confidence=0.95))
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity._f1OO352, entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        return unique_entities

    def _f1lI353(self, _f1OO352: str) -> List[str]:
        tickers = []
        for match in self.TICKER_PATTERN.finditer(_f1OO352):
            ticker = match.group(1)
            if ticker in self._TICKER_MAP or len(ticker) <= 4:
                tickers.append(ticker)
        return list(set(tickers))

class _c01I354:
    EVENT_PATTERNS: Dict[_cIIO344, List[str]] = {_cIIO344.EARNINGS: ['earnings\\s+(?:beat|miss|report)', 'quarterly\\s+results', 'revenue\\s+(?:grew|fell|rose|declined)', 'EPS\\s+(?:of|\\d)', 'profit\\s+(?:surged|plunged|rose|fell)'], _cIIO344.MERGER_ACQUISITION: ['(?:acquire|merger|acquisition|takeover|buyout)', '(?:agreed|plans)\\s+to\\s+(?:buy|acquire|merge)', '(?:all-cash|all-stock)\\s+(?:deal|offer)', 'tender\\s+offer'], _cIIO344.PRODUCT_LAUNCH: ['(?:launch|unveil|announce|introduce)\\s+(?:new|its|a)\\s+(?:product|service)', 'new\\s+(?:product|service|feature|platform)', '(?:released|rolled out|debuted)'], _cIIO344.EXECUTIVE_CHANGE: ['(?:CEO|CFO|COO|CTO)\\s+(?:steps down|resigns|retires|appointed)', '(?:names|appoints|hires)\\s+(?:new\\s+)?(?:CEO|CFO|chief)', 'executive\\s+(?:shakeup|change|departure)'], _cIIO344.REGULATORY_ACTION: ['(?:SEC|FTC|DOJ|FDA)\\s+(?:approves|rejects|investigates)', 'regulatory\\s+(?:approval|clearance|setback)', 'antitrust\\s+(?:probe|investigation|concerns)'], _cIIO344.LEGAL_ACTION: ['(?:lawsuit|sued|litigation|settlement)', '(?:court|judge)\\s+(?:rules|orders|approves)', '(?:class\\s+action|patent\\s+infringement)'], _cIIO344.ANALYST_RATING: ['(?:upgrade|downgrade|initiate|reiterate)\\s+(?:to|rating)', 'price\\s+target\\s+(?:raised|lowered|set)', '(?:buy|sell|hold|outperform|underperform)\\s+rating'], _cIIO344.DIVIDEND: ['dividend\\s+(?:increase|cut|suspend|declare)', '(?:quarterly|annual|special)\\s+dividend'], _cIIO344.STOCK_BUYBACK: ['(?:buyback|repurchase)\\s+(?:program|plan)', '(?:share|stock)\\s+repurchase', '\\$\\d+\\s*(?:billion|million)\\s+buyback'], _cIIO344.RESTRUCTURING: ['(?:layoff|workforce\\s+reduction|job\\s+cuts)', '(?:restructuring|reorganization|cost\\s+cutting)', '(?:closing|shutting\\s+down)\\s+(?:plant|facility|office)'], _cIIO344.BANKRUPTCY: ['(?:bankruptcy|chapter\\s+(?:7|11|15))', '(?:insolvency|creditor\\s+protection)', '(?:filing\\s+for|emerged\\s+from)\\s+bankruptcy'], _cIIO344.DATA_BREACH: ['(?:data\\s+breach|hack|cyberattack)', '(?:customer|user)\\s+(?:data|information)\\s+(?:exposed|stolen)', '(?:ransomware|malware|security\\s+incident)'], _cIIO344.GEOPOLITICAL: ['(?:tariff|trade\\s+war|sanctions)', '(?:geopolitical|trade)\\s+(?:tension|conflict|risk)', '(?:china|russia|iran)\\s+(?:ban|restriction)'], _cIIO344.MACRO_ECONOMIC: ['(?:fed|federal\\s+reserve)\\s+(?:rate|decision)', '(?:inflation|unemployment|GDP)\\s+(?:data|report|reading)', '(?:interest\\s+rate|monetary\\s+policy)']}

    def __init__(self):
        self._compiled_patterns: Dict[_cIIO344, List[re.Pattern]] = {}
        for event_type, patterns in self.EVENT_PATTERNS.items():
            self._compiled_patterns[event_type] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def _fOIl355(self, _f1OO352: str, _fOIO356: List[_clII347]) -> List[_c1I0348]:
        events = []
        text_lower = _f1OO352.lower()
        for event_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(text_lower)
                if match:
                    relevant_entities = [e for e in _fOIO356 if e._f1OO352.lower() in text_lower]
                    magnitude = self._estimate_magnitude(event_type, text_lower)
                    events.append(_c1I0348(event_type=event_type, description=match.group(0), entities_involved=relevant_entities[:5], confidence=0.8, magnitude=magnitude))
                    break
        return events

    def _f1lO357(self, _f0I0358: _cIIO344, _f1OO352: str) -> float:
        base_magnitudes = {_cIIO344.BANKRUPTCY: 0.95, _cIIO344.MERGER_ACQUISITION: 0.85, _cIIO344.EXECUTIVE_CHANGE: 0.7, _cIIO344.EARNINGS: 0.75, _cIIO344.REGULATORY_ACTION: 0.7, _cIIO344.LEGAL_ACTION: 0.6, _cIIO344.DATA_BREACH: 0.65, _cIIO344.RESTRUCTURING: 0.6, _cIIO344.ANALYST_RATING: 0.5, _cIIO344.PRODUCT_LAUNCH: 0.5, _cIIO344.DIVIDEND: 0.4, _cIIO344.STOCK_BUYBACK: 0.4, _cIIO344.MACRO_ECONOMIC: 0.55, _cIIO344.GEOPOLITICAL: 0.6}
        return base_magnitudes.get(_f0I0358, 0.5)

class _cIl0359:
    POSITIVE_PATTERNS = ['\\b(?:surge|soar|jump|spike|rally|boom|breakthrough|beat)\\b', '\\b(?:record|all-time high|outperform|upgrade|bullish)\\b', '\\b(?:strong|robust|impressive|stellar|exceptional)\\b', '\\b(?:growth|gain|profit|success|win|approve)\\b']
    NEGATIVE_PATTERNS = ['\\b(?:crash|plunge|tumble|slump|collapse|plummet)\\b', '\\b(?:miss|fail|loss|decline|downgrade|bearish)\\b', '\\b(?:weak|poor|disappointing|struggle|crisis)\\b', '\\b(?:lawsuit|fraud|scandal|investigation|breach)\\b', '\\b(?:layoff|bankrupt|default|warning|risk)\\b']
    AMPLIFIERS = ['\\b(?:very|extremely|significantly|sharply|dramatically)\\b', '\\b(?:massive|huge|enormous|substantial|major)\\b']

    def __init__(self):
        self._positive_patterns = [re.compile(p, re.IGNORECASE) for p in self.POSITIVE_PATTERNS]
        self._negative_patterns = [re.compile(p, re.IGNORECASE) for p in self.NEGATIVE_PATTERNS]
        self._amplifiers = [re.compile(p, re.IGNORECASE) for p in self.AMPLIFIERS]

    def _fOOl35A(self, _f1OO352: str) -> Tuple[_c0Ol346, float]:
        positive_count = sum((len(p.findall(_f1OO352)) for p in self._positive_patterns))
        negative_count = sum((len(p.findall(_f1OO352)) for p in self._negative_patterns))
        amplifier_count = sum((len(p.findall(_f1OO352)) for p in self._amplifiers))
        total = positive_count + negative_count
        if total == 0:
            return (_c0Ol346.NEUTRAL, 0.0)
        score = (positive_count - negative_count) / total
        amplification = 1 + amplifier_count * 0.1
        score *= min(amplification, 1.5)
        score = max(-1, min(1, score))
        if score >= 0.5:
            sentiment = _c0Ol346.VERY_POSITIVE
        elif score >= 0.2:
            sentiment = _c0Ol346.POSITIVE
        elif score <= -0.5:
            sentiment = _c0Ol346.VERY_NEGATIVE
        elif score <= -0.2:
            sentiment = _c0Ol346.NEGATIVE
        else:
            sentiment = _c0Ol346.NEUTRAL
        return (sentiment, score)

class _cOII35B:
    BREAKING_PATTERNS = ['\\bbreaking\\b', '\\bjust\\s+(?:now|in|announced)\\b', '\\balert\\b', '\\bflash\\b']
    URGENT_PATTERNS = ['\\burgent\\b', '\\bdeveloping\\b', '\\bimmediately\\b', '\\beffective\\s+immediately\\b']

    def __init__(self):
        self._breaking = [re.compile(p, re.IGNORECASE) for p in self.BREAKING_PATTERNS]
        self._urgent = [re.compile(p, re.IGNORECASE) for p in self.URGENT_PATTERNS]

    def _f00I35c(self, _f1OO352: str, _fOO135d: Optional[datetime]=None) -> _cI1I345:
        for pattern in self._breaking:
            if pattern.search(_f1OO352):
                return _cI1I345.BREAKING
        for pattern in self._urgent:
            if pattern.search(_f1OO352):
                return _cI1I345.URGENT
        if _fOO135d:
            age = datetime.now() - _fOO135d
            if age < timedelta(minutes=15):
                return _cI1I345.DEVELOPING
            elif age < timedelta(hours=1):
                return _cI1I345.STANDARD
        return _cI1I345.STANDARD

class _cll135E:

    def __init__(self):
        self._entity_extractor = _c0I135O()
        self._event_classifier = _c01I354()
        self._sentiment_classifier = _cIl0359()
        self._urgency_detector = _cOII35B()
        self._credibility_db = _cIII34d()
        self._registry = ComponentRegistry.get_instance()

    @bridge(connects_to=['ConditionState', 'ActivityState', 'LinguisticArbitrage', 'CorrelationCutter', 'DocumentProcessor', 'EarningsCallProcessor'], connection_types={'ConditionState': 'emits', 'ActivityState': 'triggers', 'LinguisticArbitrage': 'feeds', 'CorrelationCutter': 'signals', 'DocumentProcessor': 'extends', 'EarningsCallProcessor': 'complements'})
    def _fI1l35f(self, _fIl136O: str, _fIIl36l: str, _f10O34f: str, _fllO362: Optional[str]=None, _fI1O363: Optional[str]=None, _fOO135d: Optional[datetime]=None, _fO0O364: Optional[str]=None) -> _cI0O34A:
        full_text = f"{_fIl136O} {_fO0O364 or ''} {_fIIl36l}"
        _fOIO356 = self._entity_extractor.process_batch(full_text)
        tickers = self._entity_extractor._f1lI353(full_text)
        events = self._event_classifier._fOIl355(full_text, _fOIO356)
        sentiment, sentiment_score = self._sentiment_classifier._fOOl35A(full_text)
        urgency = self._urgency_detector._f00I35c(_fIl136O, _fOO135d)
        credibility = self._credibility_db._fOO134E(_f10O34f)
        relevance = self._calculate_relevance(entities=_fOIO356, events=events, urgency=urgency, credibility=credibility)
        return _cI0O34A(article_id='', url=_fllO362, title=_fIl136O, summary=_fO0O364 or '', body=_fIIl36l, source=_f10O34f, source_type=credibility.source_type, author=_fI1O363, published_at=_fOO135d, entities=_fOIO356, events=events, tickers_mentioned=tickers, sentiment=sentiment, sentiment_score=sentiment_score, urgency=urgency, relevance_score=relevance, credibility_score=credibility.overall_score)

    def _fI1O365(self, _fOIO356: List[_clII347], _f0I1366: List[_c1I0348], _f0lO367: _cI1I345, _f1Il368: _cIlO349) -> float:
        score = 0.3
        company_count = sum((1 for e in _fOIO356 if e.entity_type == 'COMPANY'))
        ticker_count = sum((1 for e in _fOIO356 if e.entity_type == 'TICKER'))
        score += min(0.2, (company_count + ticker_count) * 0.05)
        if _f0I1366:
            max_magnitude = max((e.magnitude for e in _f0I1366))
            score += max_magnitude * 0.25
        score += _f0lO367.value * 0.05
        score += _f1Il368.overall_score * 0.1
        return min(1.0, score)

class _c1IO369:

    def __init__(self, _fOlO36A: float=0.6):
        self._articles: Dict[str, _cI0O34A] = {}
        self._clusters: Dict[str, _cIOO34B] = {}
        self._ticker_index: Dict[str, List[str]] = defaultdict(list)
        self._similarity_threshold = _fOlO36A

    def _fIOl36B(self, _f10O36c: _cI0O34A) -> Optional[str]:
        self._articles[_f10O36c.article_id] = _f10O36c
        for ticker in _f10O36c.tickers_mentioned:
            self._ticker_index[ticker].append(_f10O36c.article_id)
        return self._cluster_article(_f10O36c)

    def _fO1l36d(self, _f10O36c: _cI0O34A) -> Optional[str]:
        similar_cluster = None
        max_similarity = 0.0
        for cluster_id, cluster in self._clusters.items():
            similarity = self._calculate_cluster_similarity(_f10O36c, cluster)
            if similarity > max_similarity and similarity >= self._similarity_threshold:
                max_similarity = similarity
                similar_cluster = cluster
        if similar_cluster:
            similar_cluster.articles.append(_f10O36c)
            similar_cluster.last_updated = datetime.now()
            similar_cluster.is_developing = True
            sentiments = [a.sentiment_score for a in similar_cluster.articles]
            similar_cluster.aggregate_sentiment = sum(sentiments) / len(sentiments)
            return similar_cluster.cluster_id
        else:
            cluster_id = f'cluster_{_f10O36c.article_id}'
            cluster = _cIOO34B(cluster_id=cluster_id, topic=self._extract_topic(_f10O36c), articles=[_f10O36c], primary_entities=_f10O36c._fOIO356[:3], primary_event=_f10O36c._f0I1366[0] if _f10O36c._f0I1366 else None, aggregate_sentiment=_f10O36c.sentiment_score, source_diversity=1.0, first_reported=_f10O36c._fOO135d, last_updated=datetime.now(), is_developing=_f10O36c._f0lO367.value >= _cI1I345.DEVELOPING.value)
            self._clusters[cluster_id] = cluster
            return cluster_id

    def _fO1O36E(self, _f10O36c: _cI0O34A, _f1l136f: _cIOO34B) -> float:
        article_entities = {e._f1OO352.lower() for e in _f10O36c._fOIO356}
        cluster_entities = {e._f1OO352.lower() for e in _f1l136f.primary_entities}
        entity_overlap = len(article_entities & cluster_entities) / max(len(article_entities | cluster_entities), 1)
        article_tickers = set(_f10O36c.tickers_mentioned)
        cluster_tickers = {t for a in _f1l136f.articles for t in a.tickers_mentioned}
        ticker_overlap = len(article_tickers & cluster_tickers) / max(len(article_tickers | cluster_tickers), 1)
        article_events = {e._f0I0358 for e in _f10O36c._f0I1366}
        cluster_events = {e._f0I0358 for a in _f1l136f.articles for e in a._f0I1366}
        event_match = len(article_events & cluster_events) / max(len(article_events | cluster_events), 1) if article_events and cluster_events else 0.5
        return entity_overlap * 0.4 + ticker_overlap * 0.35 + event_match * 0.25

    def _fOI137O(self, _f10O36c: _cI0O34A) -> str:
        if _f10O36c._f0I1366:
            return _f10O36c._f0I1366[0]._f0I0358.value
        if _f10O36c.tickers_mentioned:
            return _f10O36c.tickers_mentioned[0]
        if _f10O36c._fOIO356:
            return _f10O36c._fOIO356[0]._f1OO352
        return 'general'

    def _fO0037l(self, _f0l1372: str, _f0IO373: int=24) -> List[_cI0O34A]:
        cutoff = datetime.now() - timedelta(hours=_f0IO373)
        article_ids = self._ticker_index.get(_f0l1372, [])
        articles = []
        for aid in article_ids:
            _f10O36c = self._articles.get(aid)
            if _f10O36c and _f10O36c._fOO135d and (_f10O36c._fOO135d > cutoff):
                articles.append(_f10O36c)
        return sorted(articles, key=lambda a: a._fOO135d or datetime.min, reverse=True)

    def _f1Ol374(self) -> List[_cIOO34B]:
        return [c for c in self._clusters.values() if c.is_developing]

class _c10I375:

    def __init__(self, _fOl0376: _c1IO369):
        self._aggregator = _fOl0376

    @bridge(connects_to=['CorrelationCutter', 'ActivityState', 'OrderManager'], connection_types={'CorrelationCutter': 'feeds', 'ActivityState': 'triggers', 'OrderManager': 'signals'})
    def _flII377(self, _f0l1372: str, _fl10378: int=4) -> List[_cOOO34c]:
        articles = self._aggregator._fO0037l(_f0l1372, _fl10378)
        if not articles:
            return []
        signals = []
        sentiments = [a.sentiment_score for a in articles]
        avg_sentiment = sum(sentiments) / len(sentiments)
        if abs(avg_sentiment) > 0.3:
            direction = 'long' if avg_sentiment > 0 else 'short'
            strength = min(1.0, abs(avg_sentiment) / 0.7)
            sources = len(set((a._f10O34f for a in articles)))
            confidence = min(0.9, 0.3 + sources * 0.15 + len(articles) * 0.05)
            max_urgency = max((a._f0lO367 for a in articles))
            signals.append(_cOOO34c(signal_id=f'news_{_f0l1372}_{datetime.now().timestamp()}', signal_type='NEWS_SENTIMENT', ticker=_f0l1372, direction=direction, strength=strength, confidence=confidence, urgency=max_urgency, source_articles=[a.article_id for a in articles], expires_at=datetime.now() + timedelta(hours=2), rationale=f'Aggregate sentiment from {len(articles)} sources: {avg_sentiment:.2f}'))
        for _f10O36c in articles:
            for event in _f10O36c._f0I1366:
                if event.magnitude >= 0.7:
                    direction = self._event_to_direction(event)
                    signals.append(_cOOO34c(signal_id=f'event_{_f0l1372}_{event._f0I0358.value}_{datetime.now().timestamp()}', signal_type=f'EVENT_{event._f0I0358.value.upper()}', ticker=_f0l1372, direction=direction, strength=event.magnitude, confidence=event.confidence * _f10O36c.credibility_score, urgency=_f10O36c._f0lO367, source_articles=[_f10O36c.article_id], expires_at=datetime.now() + timedelta(hours=1), rationale=f'{event._f0I0358.value}: {event.description}'))
        return signals

    def _fIOI379(self, _flOI37A: _c1I0348) -> str:
        positive_events = {_cIIO344.EARNINGS, _cIIO344.PRODUCT_LAUNCH, _cIIO344.PARTNERSHIP, _cIIO344.EXPANSION, _cIIO344.DIVIDEND, _cIIO344.STOCK_BUYBACK}
        negative_events = {_cIIO344.BANKRUPTCY, _cIIO344.RESTRUCTURING, _cIIO344.LEGAL_ACTION, _cIIO344.DATA_BREACH, _cIIO344.REGULATORY_ACTION}
        if _flOI37A._f0I0358 in positive_events:
            return 'long'
        elif _flOI37A._f0I0358 in negative_events:
            return 'short'
        return 'neutral'

def _flOI37B() -> _cll135E:
    return _cll135E()

def _f1II37c() -> _c1IO369:
    return _c1IO369()

def _f1ll37d(_fOl0376: _c1IO369) -> _c10I375:
    return _c10I375(_fOl0376)