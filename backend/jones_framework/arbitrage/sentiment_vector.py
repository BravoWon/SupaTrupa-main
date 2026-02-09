from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import re
import numpy as np

class NarrativeType(Enum):
    CONSENSUS = auto()
    SHADOW = auto()
    NEUTRAL = auto()

@dataclass
class TextDocument:
    content: str
    source: str
    timestamp: datetime
    narrative_type: NarrativeType = NarrativeType.NEUTRAL
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SentimentVector:
    fear: float = 0.0
    distrust: float = 0.0
    divergence: float = 0.0
    urgency: float = 0.0
    contagion: float = 0.0
    timestamp: Optional[datetime] = None

    def _to_array(self) -> np.ndarray:
        return np.array([self.fear, self.distrust, self.divergence, self.urgency, self.contagion])

    @property
    def _magnitude(self) -> float:
        return float(np.linalg.norm(self._to_array()))

    @property
    def _is_significant(self) -> bool:
        return self._magnitude > 0.5

    def __add__(self, _fl1I5f8: SentimentVector) -> SentimentVector:
        return SentimentVector(fear=self.fear + _fl1I5f8.fear, distrust=self.distrust + _fl1I5f8.distrust, divergence=self.divergence + _fl1I5f8.divergence, urgency=self.urgency + _fl1I5f8.urgency, contagion=self.contagion + _fl1I5f8.contagion, timestamp=self.timestamp or _fl1I5f8.timestamp)

    def distance_to(self, _fOII5fA: float) -> SentimentVector:
        return SentimentVector(fear=self.fear * _fOII5fA, distrust=self.distrust * _fOII5fA, divergence=self.divergence * _fOII5fA, urgency=self.urgency * _fOII5fA, contagion=self.contagion * _fOII5fA, timestamp=self.timestamp)

class KeywordExtractor:
    FEAR_KEYWORDS = {'high': ['repo', 'margin call', 'contagion', 'liquidity trap', 'bank run', 'systemic risk', 'credit freeze', 'deleveraging', 'forced selling', 'circuit breaker', 'flash crash', 'counterparty risk', 'insolvency', 'lehman', 'credit crunch', 'panic selling', 'vix spike'], 'medium': ['volatility', 'risk off', 'flight to safety', 'haven assets', 'stress test', 'capital requirements', 'margin requirements', 'collateral', 'haircut', 'yield curve', 'inversion'], 'low': ['caution', 'uncertainty', 'concern', 'monitoring', 'watching']}
    DISTRUST_KEYWORDS = {'high': ['fed credibility', 'central bank failure', 'policy error', 'manipulation', 'rigged', 'fake', 'ponzi', 'bubble', 'moral hazard', 'bailout', 'too big to fail', 'corruption'], 'medium': ['intervention', 'manipulation', 'jawboning', 'forward guidance', 'credibility', 'transparency', 'accountability'], 'low': ['skeptical', 'questioning', 'doubt', 'uncertain policy']}
    URGENCY_KEYWORDS = {'high': ['immediately', 'emergency', 'urgent', 'breaking', 'crisis', 'collapse', 'plunge', 'crash', 'implode', 'meltdown'], 'medium': ['rapidly', 'quickly', 'accelerating', 'escalating', 'surging'], 'low': ['soon', 'eventually', 'gradually', 'trend']}
    CONTAGION_KEYWORDS = {'high': ['spillover', 'contagion', 'domino effect', 'cascade', 'chain reaction', 'systemic', 'interconnected', 'correlated', 'synchronized'], 'medium': ['spreading', 'affecting', 'impacting', 'ripple effect'], 'low': ['related', 'connected', 'linked']}

    def _extract_keywords(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()

        def _f0OO5fE(_f1lO5ff: Dict[str, List[str]]) -> float:
            score = 0.0
            for level, keywords in _f1lO5ff.items():
                weight = {'high': 1.0, 'medium': 0.5, 'low': 0.2}[level]
                for keyword in keywords:
                    if keyword in text_lower:
                        score += weight
            return min(1.0, score / 5.0)
        return {'fear': _f0OO5fE(self.FEAR_KEYWORDS), 'distrust': _f0OO5fE(self.DISTRUST_KEYWORDS), 'urgency': _f0OO5fE(self.URGENCY_KEYWORDS), 'contagion': _f0OO5fE(self.CONTAGION_KEYWORDS)}

class SentimentVectorPipeline:

    def __init__(self, use_transformer: bool=False, model_name: str='distilbert-base-uncased-finetuned-sst-2-english'):
        self.keyword_extractor = KeywordExtractor()
        self.use_transformer = use_transformer
        self._transformer = None
        if use_transformer:
            self._init_transformer(model_name)
        self.consensus_sources = {'bloomberg', 'reuters', 'wsj', 'ft', 'fed', 'ecb', 'boe', 'cnbc', 'marketwatch', 'barrons', 'official'}
        self.shadow_sources = {'zerohedge', 'zerohedge', 'taleb', 'burry', 'schiff', 'doomberg', 'contrarian', 'alternative', 'independent'}
        self._consensus_history: List[SentimentVector] = []
        self._shadow_history: List[SentimentVector] = []

    def _init_transformer(self, model_name: str):
        try:
            from transformers import pipeline
            self._transformer = pipeline('sentiment-analysis', model=model_name, device=-1)
        except ImportError:
            print('Warning: transformers not installed, using keyword-only mode')
            self.use_transformer = False

    def _classify_source(self, source: str) -> NarrativeType:
        source_lower = source.lower()
        for keyword in self.consensus_sources:
            if keyword in source_lower:
                return NarrativeType.CONSENSUS
        for keyword in self.shadow_sources:
            if keyword in source_lower:
                return NarrativeType.SHADOW
        return NarrativeType.NEUTRAL

    def _clean_text(self, text: str) -> str:
        text = re.sub('http\\S+', '', text)
        text = re.sub('[^\\w\\s\\.\\,\\!\\?]', '', text)
        text = ' '.join(text.split())
        return text

    def _analyze_document(self, document: TextDocument) -> SentimentVector:
        text = self._clean_text(document.content)
        keyword_scores = self.keyword_extractor._extract_keywords(text)
        ml_sentiment = 0.0
        if self.use_transformer and self._transformer:
            try:
                result = self._transformer(text[:512])[0]
                if result['label'] == 'NEGATIVE':
                    ml_sentiment = result['score']
                else:
                    ml_sentiment = 1 - result['score']
            except:
                pass
        combined_fear = (keyword_scores['fear'] + ml_sentiment) / 2 if self.use_transformer else keyword_scores['fear']
        return SentimentVector(fear=combined_fear, distrust=keyword_scores['distrust'], divergence=0.0, urgency=keyword_scores['urgency'], contagion=keyword_scores['contagion'], timestamp=document.timestamp)

    def _process_batch(self, documents: List[TextDocument]) -> Tuple[SentimentVector, Dict[str, Any]]:
        consensus_vectors = []
        shadow_vectors = []
        neutral_vectors = []
        for document in documents:
            if document.narrative_type == NarrativeType.NEUTRAL:
                document.narrative_type = self._classify_source(document.source)
            vector = self._analyze_document(document)
            if document.narrative_type == NarrativeType.CONSENSUS:
                consensus_vectors.append(vector)
                self._consensus_history.append(vector)
            elif document.narrative_type == NarrativeType.SHADOW:
                shadow_vectors.append(vector)
                self._shadow_history.append(vector)
            else:
                neutral_vectors.append(vector)

        def _aggregate_vectors(_fl1l6Oc: List[SentimentVector]) -> SentimentVector:
            if not _fl1l6Oc:
                return SentimentVector()
            result = SentimentVector()
            for v in _fl1l6Oc:
                result = result + v
            return result.distance_to(1.0 / len(_fl1l6Oc))
        consensus_agg = _aggregate_vectors(consensus_vectors)
        shadow_agg = _aggregate_vectors(shadow_vectors)
        divergence = self._calculate_divergence(consensus_agg, shadow_agg)
        all_vectors = consensus_vectors + shadow_vectors + neutral_vectors
        final = _aggregate_vectors(all_vectors)
        final.divergence = divergence
        details = {'num_documents': len(documents), 'consensus_count': len(consensus_vectors), 'shadow_count': len(shadow_vectors), 'consensus_fear': consensus_agg.fear, 'shadow_fear': shadow_agg.fear, 'divergence': divergence}
        return (final, details)

    def _calculate_divergence(self, _fll06OE: SentimentVector, _fllO6Of: SentimentVector) -> float:
        c = _fll06OE._to_array()
        s = _fllO6Of._to_array()
        norm_c = np.linalg.norm(c)
        norm_s = np.linalg.norm(s)
        if norm_c < 1e-10 or norm_s < 1e-10:
            return 0.0
        cosine_sim = np.dot(c, s) / (norm_c * norm_s)
        divergence = 1 - cosine_sim
        magnitude_diff = abs(norm_c - norm_s) / max(norm_c, norm_s, 1e-10)
        return (divergence + magnitude_diff) / 2

    def _get_regime_stress(self) -> float:
        if not self._consensus_history or not self._shadow_history:
            return 0.0
        window = 10
        recent_consensus = self._consensus_history[-window:]
        recent_shadow = self._shadow_history[-window:]
        avg_divergence = sum((self._calculate_divergence(c, s) for c, s in zip(recent_consensus, recent_shadow))) / min(len(recent_consensus), len(recent_shadow))
        avg_fear = (sum((v.fear for v in recent_consensus)) / len(recent_consensus) + sum((v.fear for v in recent_shadow)) / len(recent_shadow)) / 2
        return min(1.0, (avg_divergence + avg_fear) / 2)

    def _reset_history(self):
        self._consensus_history = []
        self._shadow_history = []