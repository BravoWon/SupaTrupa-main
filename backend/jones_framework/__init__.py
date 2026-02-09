__version__ = '0.2.0'
__author__ = 'Jones Framework Team'

# Core imports (always available)
from jones_framework.core import bridge, ComponentRegistry
from jones_framework.core.condition_state import ConditionState
from jones_framework.core.activity_state import ActivityState, RegimeID
from jones_framework.core.shadow_tensor import ShadowTensorBuilder
from jones_framework.perception.tda_pipeline import TDAPipeline
from jones_framework.perception.regime_classifier import RegimeClassifier
from jones_framework.perception.metric_warper import MetricWarper, ValueFunction
from jones_framework.sans.mixture_of_experts import MixtureOfExperts
from jones_framework.sans.lora_adapter import LoRAAdapter
from jones_framework.sans.continuity_guard import ContinuityGuard
from jones_framework.arbitrage.sentiment_vector import SentimentVectorPipeline
from jones_framework.arbitrage.linguistic_arbitrage import LinguisticArbitrageEngine

# Extended imports (may have obfuscation issues - import with fallback)
try:
    from jones_framework.trading import OrderSide, OrderType, OrderStatus, Order, Fill, Position, OrderManager
except ImportError:
    OrderSide = OrderType = OrderStatus = Order = Fill = Position = OrderManager = None

try:
    from jones_framework.strategies import StrategyType, StrategyState, Strategy, MomentumStrategy, MeanReversionStrategy, StrategyManager, create_momentum_strategy, create_mean_reversion_strategy
except ImportError:
    StrategyType = StrategyState = Strategy = MomentumStrategy = MeanReversionStrategy = StrategyManager = None
    create_momentum_strategy = create_mean_reversion_strategy = None

try:
    from jones_framework.execution import AlgorithmType, TWAPAlgorithm, VWAPAlgorithm, ExecutionManager, create_twap_algorithm, create_vwap_algorithm, create_execution_manager
except ImportError:
    AlgorithmType = TWAPAlgorithm = VWAPAlgorithm = ExecutionManager = None
    create_twap_algorithm = create_vwap_algorithm = create_execution_manager = None

try:
    from jones_framework.simulation import SimulationMode, PaperTradingEngine, BacktestEngine, MonteCarloSimulator, StressTestSimulator, create_paper_engine, create_backtest_engine
except ImportError:
    SimulationMode = PaperTradingEngine = BacktestEngine = MonteCarloSimulator = StressTestSimulator = None
    create_paper_engine = create_backtest_engine = None

try:
    from jones_framework.events import EventType, Event, EventStore, AuditLogger, EventSourcingService, create_event_sourcing_service
except ImportError:
    EventType = Event = EventStore = AuditLogger = EventSourcingService = create_event_sourcing_service = None

try:
    from jones_framework.api import HttpMethod, HttpStatus, Request, Response, Router, APIServer, WebSocketManager, create_api_server
except ImportError:
    HttpMethod = HttpStatus = Request = Response = Router = APIServer = WebSocketManager = create_api_server = None

try:
    from jones_framework.config import Environment, ConfigManager, FeatureFlagManager, SecretManager, create_config_manager, create_feature_flag_manager
except ImportError:
    Environment = ConfigManager = FeatureFlagManager = SecretManager = None
    create_config_manager = create_feature_flag_manager = None

try:
    from jones_framework.cache import EvictionPolicy, MemoryCache, QueryCache, MarketDataCache, memoize, cached, create_memory_cache, create_query_cache
except ImportError:
    EvictionPolicy = MemoryCache = QueryCache = MarketDataCache = None
    memoize = cached = create_memory_cache = create_query_cache = None

try:
    from jones_framework.scheduler import JobStatus, Job, Scheduler, TradingScheduler, RateLimiter, create_scheduler, create_trading_scheduler
except ImportError:
    JobStatus = Job = Scheduler = TradingScheduler = RateLimiter = None
    create_scheduler = create_trading_scheduler = None

try:
    from jones_framework.data.connectors import Quote, Trade, Bar, MarketDataService, create_market_data_service
except ImportError:
    Quote = Trade = Bar = MarketDataService = create_market_data_service = None

try:
    from jones_framework.ml.models import ModelType, Model, ModelRegistry, InferenceEngine, create_model_registry, create_inference_engine
except ImportError:
    ModelType = Model = ModelRegistry = InferenceEngine = None
    create_model_registry = create_inference_engine = None

try:
    from jones_framework.portfolio import PortfolioOptimizer, MeanVarianceOptimizer, RiskParityOptimizer, Rebalancer, create_mean_variance_optimizer, create_risk_parity_optimizer
except ImportError:
    PortfolioOptimizer = MeanVarianceOptimizer = RiskParityOptimizer = Rebalancer = None
    create_mean_variance_optimizer = create_risk_parity_optimizer = None

try:
    from jones_framework.monitoring import MetricType, Counter, Gauge, Histogram, MetricsRegistry, AlertManager, MonitoringService, create_monitoring_service
except ImportError:
    MetricType = Counter = Gauge = Histogram = MetricsRegistry = AlertManager = MonitoringService = create_monitoring_service = None

try:
    from jones_framework.risk import RiskEngine, RiskMetrics
except ImportError:
    RiskEngine = RiskMetrics = None

try:
    from jones_framework.features import FeatureStore
except ImportError:
    FeatureStore = None

try:
    from jones_framework.streaming import StreamProcessor
except ImportError:
    StreamProcessor = None

__all__ = ['__version__', '__author__', 'bridge', 'ComponentRegistry', 'ConditionState', 'ActivityState', 'RegimeID', 'ShadowTensorBuilder', 'TDAPipeline', 'RegimeClassifier', 'MetricWarper', 'ValueFunction', 'MixtureOfExperts', 'LoRAAdapter', 'ContinuityGuard', 'SentimentVectorPipeline', 'LinguisticArbitrageEngine', 'OrderSide', 'OrderType', 'OrderStatus', 'Order', 'Fill', 'Position', 'OrderManager', 'StrategyType', 'StrategyState', 'Strategy', 'MomentumStrategy', 'MeanReversionStrategy', 'StrategyManager', 'create_momentum_strategy', 'create_mean_reversion_strategy', 'AlgorithmType', 'TWAPAlgorithm', 'VWAPAlgorithm', 'ExecutionManager', 'create_twap_algorithm', 'create_vwap_algorithm', 'create_execution_manager', 'SimulationMode', 'PaperTradingEngine', 'BacktestEngine', 'MonteCarloSimulator', 'StressTestSimulator', 'create_paper_engine', 'create_backtest_engine', 'EventType', 'Event', 'EventStore', 'AuditLogger', 'EventSourcingService', 'create_event_sourcing_service', 'HttpMethod', 'HttpStatus', 'Request', 'Response', 'Router', 'APIServer', 'WebSocketManager', 'create_api_server', 'Environment', 'ConfigManager', 'FeatureFlagManager', 'SecretManager', 'create_config_manager', 'create_feature_flag_manager', 'EvictionPolicy', 'MemoryCache', 'QueryCache', 'MarketDataCache', 'memoize', 'cached', 'create_memory_cache', 'create_query_cache', 'JobStatus', 'Job', 'Scheduler', 'TradingScheduler', 'RateLimiter', 'create_scheduler', 'create_trading_scheduler', 'Quote', 'Trade', 'Bar', 'MarketDataService', 'create_market_data_service', 'ModelType', 'Model', 'ModelRegistry', 'InferenceEngine', 'create_model_registry', 'create_inference_engine', 'PortfolioOptimizer', 'MeanVarianceOptimizer', 'RiskParityOptimizer', 'Rebalancer', 'create_mean_variance_optimizer', 'create_risk_parity_optimizer', 'MetricType', 'Counter', 'Gauge', 'Histogram', 'MetricsRegistry', 'AlertManager', 'MonitoringService', 'create_monitoring_service', 'RiskEngine', 'RiskMetrics', 'FeatureStore', 'StreamProcessor']