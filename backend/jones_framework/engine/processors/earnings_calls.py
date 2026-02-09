from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum, auto
from datetime import datetime
import re
import math
from collections import defaultdict
from jones_framework.core import bridge, ComponentRegistry

class _c0l03O5(Enum):
    CEO = 'ceo'
    CFO = 'cfo'
    COO = 'coo'
    CTO = 'cto'
    PRESIDENT = 'president'
    IR_OFFICER = 'ir_officer'
    BOARD_MEMBER = 'board_member'
    OTHER_EXECUTIVE = 'other_executive'
    ANALYST = 'analyst'
    OPERATOR = 'operator'
    UNKNOWN = 'unknown'

class _cOlO3O6(Enum):
    FORWARD_LOOKING = auto()
    HISTORICAL = auto()
    GUIDANCE = auto()
    DISCLAIMER = auto()
    QUESTION = auto()
    RESPONSE = auto()
    TRANSITION = auto()

class _cO1l3O7(Enum):
    CONFIDENT = 'confident'
    CAUTIOUS = 'cautious'
    DEFENSIVE = 'defensive'
    OPTIMISTIC = 'optimistic'
    PESSIMISTIC = 'pessimistic'
    NEUTRAL = 'neutral'
    EVASIVE = 'evasive'
    TRANSPARENT = 'transparent'

@dataclass
class _c1I03O8:
    name: str
    role: _c0l03O5
    company: Optional[str] = None
    firm: Optional[str] = None
    title: Optional[str] = None
    statement_count: int = 0
    word_count: int = 0
    avg_sentiment: float = 0.0
    tone_distribution: Dict[_cO1l3O7, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.tone_distribution:
            self.tone_distribution = {cat: 0.0 for cat in _cO1l3O7}

@dataclass
class _clI03O9:
    speaker: _c1I03O8
    text: str
    timestamp: Optional[str] = None
    statement_type: _cOlO3O6 = _cOlO3O6.HISTORICAL
    sentiment_score: float = 0.0
    confidence_score: float = 0.0
    tone: _cO1l3O7 = _cO1l3O7.NEUTRAL
    entities: List[str] = field(default_factory=list)
    metrics_mentioned: List[Dict[str, Any]] = field(default_factory=list)
    forward_looking_phrases: List[str] = field(default_factory=list)
    hedging_phrases: List[str] = field(default_factory=list)

@dataclass
class _cl113OA:
    analyst: _c1I03O8
    question: _clI03O9
    respondents: List[_c1I03O8]
    responses: List[_clI03O9]
    follow_ups: List[_clI03O9] = field(default_factory=list)
    topic: Optional[str] = None
    sentiment_delta: float = 0.0
    evasiveness_score: float = 0.0

@dataclass
class _cl0l3OB:
    topic: str
    keywords: List[str]
    statements: List[_clI03O9]
    avg_sentiment: float = 0.0
    sentiment_trajectory: List[float] = field(default_factory=list)
    speaker_distribution: Dict[str, int] = field(default_factory=dict)

@dataclass
class _cOI03Oc:
    signal_type: str
    description: str
    strength: float
    confidence: float
    evidence: List[str]
    historical_correlation: Optional[float] = None

@dataclass
class _cl1l3Od:
    company: str
    ticker: str
    date: datetime
    quarter: str
    fiscal_year: int
    speakers: List[_c1I03O8]
    management_team: List[_c1I03O8]
    analysts: List[_c1I03O8]
    prepared_remarks: List[_clI03O9]
    qa_session: List[_cl113OA]
    overall_sentiment: float
    sentiment_trajectory: List[float]
    management_tone: Dict[_cO1l3O7, float]
    topic_clusters: List[_cl0l3OB]
    linguistic_signals: List[_cOI03Oc]
    forward_looking_ratio: float
    hedging_ratio: float
    specificity_score: float
    evasiveness_score: float
    guidance_statements: List[_clI03O9]
    guidance_changes: Dict[str, str]
    yoy_tone_change: Optional[float] = None
    qoq_tone_change: Optional[float] = None
    raw_transcript: str = ''
    word_count: int = 0
    duration_minutes: Optional[float] = None

class _cO0l3OE:
    POSITIVE_WORDS = {'strong', 'growth', 'exceeded', 'beat', 'outperformed', 'record', 'momentum', 'optimistic', 'confident', 'pleased', 'excellent', 'robust', 'healthy', 'accelerating', 'improved', 'profitable', 'expanding', 'successful', 'opportunity', 'innovative', 'leading', 'exceptional', 'outstanding', 'remarkable', 'significant', 'solid', 'upside', 'tailwind', 'strength', 'surge', 'boom', 'thriving'}
    NEGATIVE_WORDS = {'weak', 'decline', 'missed', 'challenging', 'headwind', 'pressure', 'uncertain', 'concerned', 'disappointing', 'struggled', 'difficult', 'slower', 'reduced', 'impacted', 'volatile', 'risk', 'loss', 'decreased', 'deteriorated', 'delayed', 'underperformed', 'shortfall', 'downturn', 'contraction', 'setback', 'obstacle', 'hurdle'}
    HEDGING_PHRASES = ['we believe', 'we expect', 'we anticipate', 'we think', 'it is possible', 'may be', 'might be', 'could be', 'subject to', 'depending on', 'contingent upon', 'to some extent', 'in some cases', 'generally speaking', 'for the most part', 'in many ways', 'relatively', 'somewhat', 'fairly', 'pretty much', 'more or less']
    FORWARD_LOOKING_PHRASES = ['going forward', 'in the future', 'next quarter', 'next year', 'we expect', 'we anticipate', 'we project', 'we forecast', 'our guidance', 'our outlook', 'we plan to', 'we intend to', 'looking ahead', 'in the coming', 'our target', 'we aim to', 'pipeline', 'roadmap', 'trajectory', 'on track to']
    EVASIVE_PATTERNS = ["that's a (great|good) question", 'as (i|we) mentioned (earlier|before)', "i (don't|do not) want to get into (specifics|details)", "we'll have more to share", "it's (too early|premature) to", "we're still (evaluating|assessing|analyzing)", "i can't (speak|comment) to that", "that's (really|kind of) a (board|strategic) (decision|matter)"]
    CONFIDENT_PHRASES = ['we are confident', 'we are certain', 'clearly', 'without a doubt', 'definitely', 'absolutely', 'we are committed', 'we will', 'we are going to', 'make no mistake', 'unequivocally', 'undoubtedly']

    def __init__(self):
        self._word_cache: Dict[str, float] = {}
        self._compile_patterns()

    def _fll13Of(self):
        self._hedging_patterns = [re.compile(phrase, re.IGNORECASE) for phrase in self.HEDGING_PHRASES]
        self._forward_patterns = [re.compile(phrase, re.IGNORECASE) for phrase in self.FORWARD_LOOKING_PHRASES]
        self._evasive_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.EVASIVE_PATTERNS]
        self._confident_patterns = [re.compile(phrase, re.IGNORECASE) for phrase in self.CONFIDENT_PHRASES]

    def process_batch(self, _f0OO3ll: str) -> Tuple[float, float]:
        words = _f0OO3ll.lower().split()
        if not words:
            return (0.0, 0.0)
        positive_count = 0
        negative_count = 0
        for word in words:
            clean_word = re.sub('[^\\w]', '', word)
            if clean_word in self.POSITIVE_WORDS:
                positive_count += 1
            elif clean_word in self.NEGATIVE_WORDS:
                negative_count += 1
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return (0.0, 0.0)
        sentiment = (positive_count - negative_count) / total_sentiment_words
        confidence = min(1.0, total_sentiment_words / (len(words) * 0.1))
        return (sentiment, confidence)

    def _fl0O3l2(self, _f0OO3ll: str) -> List[str]:
        found = []
        for pattern in self._hedging_patterns:
            matches = pattern.findall(_f0OO3ll)
            found.extend(matches)
        return found

    def _f0OO3l3(self, _f0OO3ll: str) -> List[str]:
        found = []
        for pattern in self._forward_patterns:
            matches = pattern.findall(_f0OO3ll)
            found.extend(matches)
        return found

    def _f10I3l4(self, _f0OO3ll: str) -> float:
        evasive_count = sum((1 for pattern in self._evasive_patterns if pattern.search(_f0OO3ll)))
        return min(1.0, evasive_count / 3)

    def _f0lI3l5(self, _f0OO3ll: str) -> float:
        confident_count = sum((1 for pattern in self._confident_patterns if pattern.search(_f0OO3ll)))
        return min(1.0, confident_count / 2)

    def _f1013l6(self, _f0OO3ll: str) -> _cO1l3O7:
        sentiment, _ = self.process_batch(_f0OO3ll)
        hedging = len(self._fl0O3l2(_f0OO3ll))
        evasiveness = self._f10I3l4(_f0OO3ll)
        confidence = self._f0lI3l5(_f0OO3ll)
        if evasiveness > 0.5:
            return _cO1l3O7.EVASIVE
        if confidence > 0.5:
            if sentiment > 0.2:
                return _cO1l3O7.CONFIDENT
            else:
                return _cO1l3O7.DEFENSIVE
        if hedging > 2:
            return _cO1l3O7.CAUTIOUS
        if sentiment > 0.3:
            return _cO1l3O7.OPTIMISTIC
        elif sentiment < -0.3:
            return _cO1l3O7.PESSIMISTIC
        return _cO1l3O7.NEUTRAL

class _c1I03l7:
    SPEAKER_PATTERN = re.compile('^(?P<name>[A-Z][a-zA-Z\\s\\.\\-]+?)(?:\\s*[-–—]\\s*|\\s*,\\s*)(?P<title>[A-Za-z\\s,&]+)?$', re.MULTILINE)
    ROLE_KEYWORDS = {_c0l03O5.CEO: ['ceo', 'chief executive', 'chief exec'], _c0l03O5.CFO: ['cfo', 'chief financial', 'finance officer'], _c0l03O5.COO: ['coo', 'chief operating', 'operations officer'], _c0l03O5.CTO: ['cto', 'chief technology', 'tech officer'], _c0l03O5.PRESIDENT: ['president'], _c0l03O5.IR_OFFICER: ['investor relations', 'ir '], _c0l03O5.BOARD_MEMBER: ['chairman', 'board', 'director'], _c0l03O5.ANALYST: ['analyst'], _c0l03O5.OPERATOR: ['operator', 'conference call']}
    SECTION_MARKERS = {'prepared': ['prepared remarks', 'opening remarks', 'presentation', 'management discussion', 'let me begin', "i'd like to start"], 'qa': ['q&a', 'question-and-answer', 'question and answer', "we'll now take questions", 'open the line for questions', "operator, we're ready for questions"]}

    def __init__(self):
        self._speakers_cache: Dict[str, _c1I03O8] = {}

    def _fI103l8(self, _f0OO3ll: str) -> Tuple[List[_clI03O9], int]:
        statements = []
        lines = _f0OO3ll.split('\n')
        current_speaker: Optional[_c1I03O8] = None
        current_text_parts: List[str] = []
        section_boundary = self._find_qa_boundary(_f0OO3ll)
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            speaker = self._parse_speaker_line(line)
            if speaker:
                if current_speaker and current_text_parts:
                    statement = self._create_statement(current_speaker, ' '.join(current_text_parts), is_qa_section=i > section_boundary)
                    statements.append(statement)
                current_speaker = speaker
                current_text_parts = []
            else:
                current_text_parts.append(line)
        if current_speaker and current_text_parts:
            statement = self._create_statement(current_speaker, ' '.join(current_text_parts), is_qa_section=True)
            statements.append(statement)
        return (statements, section_boundary)

    def _f11I3l9(self, _f0OO3ll: str) -> int:
        text_lower = _f0OO3ll.lower()
        lines = _f0OO3ll.split('\n')
        for marker in self.SECTION_MARKERS['qa']:
            idx = text_lower.find(marker)
            if idx != -1:
                prefix = _f0OO3ll[:idx]
                return prefix.count('\n')
        return int(len(lines) * 0.6)

    def _f1l13lA(self, _fOI03lB: str) -> Optional[_c1I03O8]:
        match = self.SPEAKER_PATTERN.match(_fOI03lB)
        if match:
            name = match.group('name').strip()
            title = match.group('title') or ''
            role = self._infer_role(title, name)
            cache_key = name.lower()
            if cache_key in self._speakers_cache:
                return self._speakers_cache[cache_key]
            speaker = _c1I03O8(name=name, role=role, title=title)
            self._speakers_cache[cache_key] = speaker
            return speaker
        return None

    def _fl113lc(self, _fIII3ld: str, _f1103lE: str) -> _c0l03O5:
        title_lower = _fIII3ld.lower()
        name_lower = _f1103lE.lower()
        combined = f'{title_lower} {name_lower}'
        for role, keywords in self.ROLE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in combined:
                    return role
        if ',' in _fIII3ld or any((firm in title_lower for firm in ['capital', 'securities', 'research', 'partners', 'investments'])):
            return _c0l03O5.ANALYST
        return _c0l03O5.OTHER_EXECUTIVE

    def _f0IO3lf(self, _f1IO32O: _c1I03O8, _f0OO3ll: str, _f0lO32l: bool) -> _clI03O9:
        analyzer = _cO0l3OE()
        sentiment, confidence = analyzer.process_batch(_f0OO3ll)
        hedging = analyzer._fl0O3l2(_f0OO3ll)
        forward = analyzer._f0OO3l3(_f0OO3ll)
        tone = analyzer._f1013l6(_f0OO3ll)
        if _f1IO32O.role == _c0l03O5.ANALYST:
            stmt_type = _cOlO3O6.QUESTION
        elif _f0lO32l and _f1IO32O.role != _c0l03O5.OPERATOR:
            stmt_type = _cOlO3O6.RESPONSE
        elif forward:
            stmt_type = _cOlO3O6.FORWARD_LOOKING
        else:
            stmt_type = _cOlO3O6.HISTORICAL
        metrics = self._extract_metrics(_f0OO3ll)
        entities = self._extract_entities(_f0OO3ll)
        return _clI03O9(speaker=_f1IO32O, text=_f0OO3ll, statement_type=stmt_type, sentiment_score=sentiment, confidence_score=confidence, tone=tone, entities=entities, metrics_mentioned=metrics, forward_looking_phrases=forward, hedging_phrases=hedging)

    def _f1OO322(self, _f0OO3ll: str) -> List[Dict[str, Any]]:
        metrics = []
        revenue_pattern = re.compile('revenue\\s+(?:of\\s+)?(?:\\$)?(\\d+(?:\\.\\d+)?)\\s*(million|billion|M|B)?', re.IGNORECASE)
        for match in revenue_pattern.finditer(_f0OO3ll):
            value = float(match.group(1))
            unit = match.group(2) or ''
            if unit.lower() in ['billion', 'b']:
                value *= 1000000000.0
            elif unit.lower() in ['million', 'm']:
                value *= 1000000.0
            metrics.append({'name': 'revenue', 'value': value, 'context': match.group(0)})
        eps_pattern = re.compile('(?:eps|earnings per share)\\s+(?:of\\s+)?(?:\\$)?(\\d+\\.\\d+)', re.IGNORECASE)
        for match in eps_pattern.finditer(_f0OO3ll):
            metrics.append({'name': 'eps', 'value': float(match.group(1)), 'context': match.group(0)})
        margin_pattern = re.compile('(gross|operating|net|profit)\\s+margin\\s+(?:of\\s+)?(\\d+(?:\\.\\d+)?)\\s*%', re.IGNORECASE)
        for match in margin_pattern.finditer(_f0OO3ll):
            metrics.append({'name': f'{match.group(1).lower()}_margin', 'value': float(match.group(2)), 'context': match.group(0)})
        growth_pattern = re.compile('(?:grew|growth|increased|up)\\s+(?:by\\s+)?(\\d+(?:\\.\\d+)?)\\s*%', re.IGNORECASE)
        for match in growth_pattern.finditer(_f0OO3ll):
            metrics.append({'name': 'growth_rate', 'value': float(match.group(1)), 'context': match.group(0)})
        return metrics

    def _fI1I323(self, _f0OO3ll: str) -> List[str]:
        entities = []
        company_pattern = re.compile('\\b([A-Z][A-Za-z]+(?:\\s+[A-Z][A-Za-z]+)*)\\s+(?:Inc|Corp|Ltd|LLC|Company|Co)\\b')
        entities.extend(company_pattern.findall(_f0OO3ll))
        quoted = re.findall('"([^"]+)"', _f0OO3ll)
        entities.extend(quoted)
        return list(set(entities))

class _c1IO324:

    def __init__(self):
        self._parser = _c1I03l7()
        self._analyzer = _cO0l3OE()
        self._registry = ComponentRegistry.get_instance()

    @bridge(connects_to=['ConditionState', 'ActivityState', 'LinguisticArbitrage', 'CorrelationCutter', 'DocumentProcessor'], connection_types={'ConditionState': 'emits', 'ActivityState': 'triggers', 'LinguisticArbitrage': 'feeds', 'CorrelationCutter': 'signals', 'DocumentProcessor': 'extends'})
    def _f11l325(self, _fI1l326: str, _f0OI327: str, _fl1I328: str, _f001329: datetime, _f0Il32A: str='Q1', _fIO032B: int=2024) -> _cl1l3Od:
        statements, qa_boundary = self._parser._fI103l8(_fI1l326)
        prepared_remarks = [s for i, s in enumerate(statements) if i < qa_boundary and s._f1IO32O.role != _c0l03O5.OPERATOR]
        qa_statements = [s for i, s in enumerate(statements) if i >= qa_boundary]
        qa_pairs = self._extract_qa_pairs(qa_statements)
        all_speakers = list({s._f1IO32O for s in statements})
        management = [s for s in all_speakers if s.role in [_c0l03O5.CEO, _c0l03O5.CFO, _c0l03O5.COO, _c0l03O5.CTO, _c0l03O5.PRESIDENT, _c0l03O5.IR_OFFICER, _c0l03O5.OTHER_EXECUTIVE]]
        analysts = [s for s in all_speakers if s.role == _c0l03O5.ANALYST]
        all_management_statements = [s for s in statements if s._f1IO32O in management]
        forward_looking_count = sum((1 for s in all_management_statements if s.statement_type == _cOlO3O6.FORWARD_LOOKING))
        forward_looking_ratio = forward_looking_count / len(all_management_statements) if all_management_statements else 0
        hedging_count = sum((len(s.hedging_phrases) for s in all_management_statements))
        total_words = sum((len(s._f0OO3ll.split()) for s in all_management_statements))
        hedging_ratio = hedging_count / (total_words / 100) if total_words else 0
        sentiment_trajectory = [s.sentiment_score for s in statements]
        overall_sentiment = sum(sentiment_trajectory) / len(sentiment_trajectory) if sentiment_trajectory else 0
        tone_counts: Dict[_cO1l3O7, int] = defaultdict(int)
        for s in all_management_statements:
            tone_counts[s.tone] += 1
        total_mgmt = len(all_management_statements) or 1
        management_tone = {tone: count / total_mgmt for tone, count in tone_counts.items()}
        metrics_per_statement = sum((len(s.metrics_mentioned) for s in all_management_statements)) / total_mgmt if total_mgmt else 0
        specificity_score = min(1.0, metrics_per_statement / 2)
        avg_evasiveness = sum((qa.evasiveness_score for qa in qa_pairs)) / len(qa_pairs) if qa_pairs else 0
        guidance_statements = [s for s in all_management_statements if any((kw in s._f0OO3ll.lower() for kw in ['guidance', 'outlook', 'expect', 'target', 'forecast']))]
        linguistic_signals = self._detect_linguistic_signals(statements, management, qa_pairs)
        topic_clusters = self._cluster_topics(statements)
        return _cl1l3Od(company=_f0OI327, ticker=_fl1I328, date=_f001329, quarter=_f0Il32A, fiscal_year=_fIO032B, speakers=all_speakers, management_team=management, analysts=analysts, prepared_remarks=prepared_remarks, qa_session=qa_pairs, overall_sentiment=overall_sentiment, sentiment_trajectory=sentiment_trajectory, management_tone=management_tone, topic_clusters=topic_clusters, linguistic_signals=linguistic_signals, forward_looking_ratio=forward_looking_ratio, hedging_ratio=hedging_ratio, specificity_score=specificity_score, evasiveness_score=avg_evasiveness, guidance_statements=guidance_statements, guidance_changes={}, raw_transcript=_fI1l326, word_count=len(_fI1l326.split()))

    def _flOO32c(self, _f11I32d: List[_clI03O9]) -> List[_cl113OA]:
        pairs = []
        current_question: Optional[_clI03O9] = None
        current_analyst: Optional[_c1I03O8] = None
        current_responses: List[_clI03O9] = []
        current_respondents: List[_c1I03O8] = []
        for stmt in _f11I32d:
            if stmt._f1IO32O.role == _c0l03O5.ANALYST:
                if current_question and current_responses:
                    pairs.append(self._create_qa_pair(current_analyst, current_question, current_respondents, current_responses))
                current_question = stmt
                current_analyst = stmt._f1IO32O
                current_responses = []
                current_respondents = []
            elif stmt._f1IO32O.role != _c0l03O5.OPERATOR:
                current_responses.append(stmt)
                if stmt._f1IO32O not in current_respondents:
                    current_respondents.append(stmt._f1IO32O)
        if current_question and current_responses:
            pairs.append(self._create_qa_pair(current_analyst, current_question, current_respondents, current_responses))
        return pairs

    def _fI0O32E(self, _fO1132f: _c1I03O8, _flI133O: _clI03O9, _fl1I33l: List[_c1I03O8], _fOOO332: List[_clI03O9]) -> _cl113OA:
        q_sentiment = _flI133O.sentiment_score
        avg_response_sentiment = sum((r.sentiment_score for r in _fOOO332)) / len(_fOOO332) if _fOOO332 else 0
        sentiment_delta = avg_response_sentiment - q_sentiment
        evasiveness = sum((self._analyzer._f10I3l4(r._f0OO3ll) for r in _fOOO332)) / len(_fOOO332) if _fOOO332 else 0
        return _cl113OA(analyst=_fO1132f, question=_flI133O, respondents=_fl1I33l, responses=_fOOO332, sentiment_delta=sentiment_delta, evasiveness_score=evasiveness)

    def _fl0I333(self, _f0OO334: List[_clI03O9], _f1OO335: List[_c1I03O8], _fO1I336: List[_cl113OA]) -> List[_cOI03Oc]:
        signals = []
        mgmt_statements = [s for s in _f0OO334 if s._f1IO32O in _f1OO335]
        if len(mgmt_statements) >= 10:
            first_half = mgmt_statements[:len(mgmt_statements) // 2]
            second_half = mgmt_statements[len(mgmt_statements) // 2:]
            first_sentiment = sum((s.sentiment_score for s in first_half)) / len(first_half)
            second_sentiment = sum((s.sentiment_score for s in second_half)) / len(second_half)
            shift = second_sentiment - first_sentiment
            if abs(shift) > 0.2:
                signals.append(_cOI03Oc(signal_type='TONE_SHIFT', description=f"Sentiment shifted {('up' if shift > 0 else 'down')} during call", strength=min(1.0, abs(shift) / 0.5), confidence=0.7, evidence=[s._f0OO3ll[:100] for s in second_half[:2]], historical_correlation=0.65))
        prepared_hedging = sum((len(s.hedging_phrases) for s in _f0OO334 if s.statement_type != _cOlO3O6.RESPONSE))
        qa_hedging = sum((len(s.hedging_phrases) for s in _f0OO334 if s.statement_type == _cOlO3O6.RESPONSE))
        prepared_count = sum((1 for s in _f0OO334 if s.statement_type != _cOlO3O6.RESPONSE))
        qa_count = sum((1 for s in _f0OO334 if s.statement_type == _cOlO3O6.RESPONSE))
        if prepared_count > 0 and qa_count > 0:
            hedging_increase = qa_hedging / qa_count - prepared_hedging / prepared_count
            if hedging_increase > 0.5:
                signals.append(_cOI03Oc(signal_type='HEDGING_INCREASE', description='Management hedging significantly more in Q&A', strength=min(1.0, hedging_increase), confidence=0.75, evidence=[s._f0OO3ll[:100] for s in _f0OO334 if s.hedging_phrases and s.statement_type == _cOlO3O6.RESPONSE][:3]))
        if _fO1I336:
            avg_evasiveness = sum((qa.evasiveness_score for qa in _fO1I336)) / len(_fO1I336)
            if avg_evasiveness > 0.3:
                most_evasive = max(_fO1I336, key=lambda q: q.evasiveness_score)
                signals.append(_cOI03Oc(signal_type='HIGH_EVASIVENESS', description='Management being evasive in Q&A responses', strength=min(1.0, avg_evasiveness / 0.5), confidence=0.8, evidence=[r._f0OO3ll[:100] for r in most_evasive._fOOO332[:2]]))
        metrics_mentioned = sum((len(s.metrics_mentioned) for s in mgmt_statements))
        words = sum((len(s._f0OO3ll.split()) for s in mgmt_statements))
        specificity = metrics_mentioned / (words / 500) if words else 0
        if specificity < 0.5:
            signals.append(_cOI03Oc(signal_type='LOW_SPECIFICITY', description='Management providing few specific metrics', strength=1.0 - specificity, confidence=0.6, evidence=[]))
        ceo_statements = [s for s in _f0OO334 if s._f1IO32O.role == _c0l03O5.CEO]
        cfo_statements = [s for s in _f0OO334 if s._f1IO32O.role == _c0l03O5.CFO]
        if ceo_statements and cfo_statements:
            ceo_sentiment = sum((s.sentiment_score for s in ceo_statements)) / len(ceo_statements)
            cfo_sentiment = sum((s.sentiment_score for s in cfo_statements)) / len(cfo_statements)
            divergence = ceo_sentiment - cfo_sentiment
            if abs(divergence) > 0.25:
                signals.append(_cOI03Oc(signal_type='CEO_CFO_DIVERGENCE', description=f"CEO {('more' if divergence > 0 else 'less')} optimistic than CFO", strength=min(1.0, abs(divergence) / 0.5), confidence=0.7, evidence=[f'CEO: {ceo_statements[0]._f0OO3ll[:80]}...', f'CFO: {cfo_statements[0]._f0OO3ll[:80]}...']))
        return signals

    def _fOO1337(self, _f0OO334: List[_clI03O9]) -> List[_cl0l3OB]:
        topic_keywords = {'revenue': ['revenue', 'sales', 'top line', 'bookings', 'orders'], 'margins': ['margin', 'profitability', 'costs', 'expenses', 'operating'], 'growth': ['growth', 'expanding', 'accelerating', 'momentum'], 'guidance': ['guidance', 'outlook', 'forecast', 'expect', 'target'], 'competition': ['competition', 'market share', 'competitive', 'rivals'], 'innovation': ['innovation', 'r&d', 'research', 'new products', 'pipeline'], 'capital': ['capital', 'investment', 'capex', 'returns', 'dividends', 'buyback'], 'risks': ['risk', 'challenge', 'headwind', 'uncertainty', 'concern']}
        clusters: Dict[str, List[_clI03O9]] = defaultdict(list)
        for statement in _f0OO334:
            text_lower = statement._f0OO3ll.lower()
            for topic, keywords in topic_keywords.items():
                if any((kw in text_lower for kw in keywords)):
                    clusters[topic].append(statement)
        result = []
        for topic, stmts in clusters.items():
            if not stmts:
                continue
            sentiments = [s.sentiment_score for s in stmts]
            speaker_dist: Dict[str, int] = defaultdict(int)
            for s in stmts:
                speaker_dist[s._f1IO32O._f1103lE] += 1
            result.append(_cl0l3OB(topic=topic, keywords=topic_keywords[topic], statements=stmts, avg_sentiment=sum(sentiments) / len(sentiments), sentiment_trajectory=sentiments, speaker_distribution=dict(speaker_dist)))
        return result

    def _fOlI338(self, _f1lI339: _cl1l3Od) -> str:
        lines = [f'# {_f1lI339._f0OI327} ({_f1lI339._fl1I328}) Earnings Call Analysis', f"## {_f1lI339._f0Il32A} {_f1lI339._fIO032B} - {_f1lI339._f001329.strftime('%Y-%m-%d')}", '', '### Key Metrics', f'- Overall Sentiment: {_f1lI339.overall_sentiment:.2f} (-1 to 1)', f'- Forward-Looking Ratio: {_f1lI339.forward_looking_ratio:.1%}', f'- Hedging Ratio: {_f1lI339.hedging_ratio:.2f}', f'- Specificity Score: {_f1lI339.specificity_score:.2f}', f'- Evasiveness Score: {_f1lI339.evasiveness_score:.2f}', '', '### Management Tone Distribution']
        for tone, pct in sorted(_f1lI339.management_tone.items(), key=lambda x: x[1], reverse=True):
            if pct > 0:
                lines.append(f'- {tone.value._fIII3ld()}: {pct:.1%}')
        lines.extend(['', '### Linguistic Signals'])
        for signal in _f1lI339.linguistic_signals:
            lines.append(f'- **{signal.signal_type}** (strength: {signal.strength:.2f}): {signal.description}')
        if _f1lI339.topic_clusters:
            lines.extend(['', '### Topic Clusters'])
            for cluster in sorted(_f1lI339.topic_clusters, key=lambda c: len(c._f0OO334), reverse=True)[:5]:
                lines.append(f'- **{cluster.topic._fIII3ld()}** ({len(cluster._f0OO334)} mentions, sentiment: {cluster.avg_sentiment:.2f})')
        return '\n'.join(lines)

class _c0OO33A:

    def __init__(self):
        self._historical: Dict[str, List[_cl1l3Od]] = {}

    def _flll33B(self, _f1lI339: _cl1l3Od):
        key = _f1lI339._fl1I328
        if key not in self._historical:
            self._historical[key] = []
        self._historical[key].append(_f1lI339)
        self._historical[key].sort(key=lambda a: a._f001329)

    def _fI1133c(self, _fl1I328: str) -> Optional[Dict[str, Any]]:
        calls = self._historical.get(_fl1I328, [])
        if len(calls) < 2:
            return None
        current = calls[-1]
        previous = calls[-2]
        return {'ticker': _fl1I328, 'current_quarter': f'{current._f0Il32A} {current._fIO032B}', 'previous_quarter': f'{previous._f0Il32A} {previous._fIO032B}', 'sentiment_change': current.overall_sentiment - previous.overall_sentiment, 'hedging_change': current.hedging_ratio - previous.hedging_ratio, 'specificity_change': current.specificity_score - previous.specificity_score, 'evasiveness_change': current.evasiveness_score - previous.evasiveness_score, 'forward_looking_change': current.forward_looking_ratio - previous.forward_looking_ratio, 'new_signals': [s for s in current.linguistic_signals if s.signal_type not in [ps.signal_type for ps in previous.linguistic_signals]], 'resolved_signals': [s for s in previous.linguistic_signals if s.signal_type not in [cs.signal_type for cs in current.linguistic_signals]]}

    def _f0Ol33d(self, _fl1I328: str, _fl1133E: int=4) -> Optional[Dict[str, Any]]:
        calls = self._historical.get(_fl1I328, [])
        if len(calls) < _fl1133E:
            _fl1133E = len(calls)
        if _fl1133E < 2:
            return None
        recent = calls[-_fl1133E:]
        sentiments = [c.overall_sentiment for c in recent]
        hedging = [c.hedging_ratio for c in recent]

        def _fl0I33f(_f10034O: List[float]) -> float:
            n = len(_f10034O)
            if n < 2:
                return 0.0
            x_mean = (n - 1) / 2
            y_mean = sum(_f10034O) / n
            numerator = sum(((i - x_mean) * (v - y_mean) for i, v in enumerate(_f10034O)))
            denominator = sum(((i - x_mean) ** 2 for i in range(n)))
            return numerator / denominator if denominator else 0.0
        return {'ticker': _fl1I328, 'periods': _fl1133E, 'sentiment_trend': _fl0I33f(sentiments), 'hedging_trend': _fl0I33f(hedging), 'avg_sentiment': sum(sentiments) / len(sentiments), 'sentiment_volatility': (sum(((s - sum(sentiments) / len(sentiments)) ** 2 for s in sentiments)) / len(sentiments)) ** 0.5, 'latest_signals': recent[-1].linguistic_signals if recent else []}

def _fIl034l() -> _c1IO324:
    return _c1IO324()

def _flll342(_fI1l326: str, _f0OI327: str, _fl1I328: str, _f001329: Optional[datetime]=None) -> _cl1l3Od:
    processor = _c1IO324()
    return processor._f11l325(transcript=_fI1l326, company=_f0OI327, ticker=_fl1I328, date=_f001329 or datetime.now())