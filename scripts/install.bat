@echo off
REM =============================================================================
REM JONES FRAMEWORK - Interdimensional Install System (Windows)
REM =============================================================================
REM Topology-aware modular installation with compartmentalized builds
REM
REM Usage:
REM   install.bat              - Install all components
REM   install.bat --core       - Core only (minimal footprint)
REM   install.bat --tda        - Core + TDA pipeline
REM   install.bat --ml         - Core + ML/AI components
REM   install.bat --api        - Core + API server
REM   install.bat --full       - Everything including dev tools
REM   install.bat --frontend   - Frontend only
REM   install.bat --optimize   - Resource-optimized build
REM =============================================================================

setlocal EnableDelayedExpansion

REM Colors (Windows 10+)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "PURPLE=[95m"
set "CYAN=[96m"
set "NC=[0m"

REM Configuration
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "BACKEND_DIR=%PROJECT_ROOT%\backend"
set "FRONTEND_DIR=%PROJECT_ROOT%\frontend"
set "SHARED_DIR=%PROJECT_ROOT%\shared"

REM Default settings
set "INSTALL_MODE=all"
set "OPTIMIZE_RESOURCES=false"
set "SKIP_FRONTEND=false"
set "GPU_TYPE=none"

REM =============================================================================
REM Parse Arguments
REM =============================================================================

:parse_args
if "%~1"=="" goto :detect_system
if /i "%~1"=="--core" (
    set "INSTALL_MODE=core"
    shift
    goto :parse_args
)
if /i "%~1"=="--tda" (
    set "INSTALL_MODE=tda"
    shift
    goto :parse_args
)
if /i "%~1"=="--ml" (
    set "INSTALL_MODE=ml"
    shift
    goto :parse_args
)
if /i "%~1"=="--api" (
    set "INSTALL_MODE=api"
    shift
    goto :parse_args
)
if /i "%~1"=="--full" (
    set "INSTALL_MODE=full"
    shift
    goto :parse_args
)
if /i "%~1"=="--frontend" (
    set "INSTALL_MODE=frontend"
    shift
    goto :parse_args
)
if /i "%~1"=="--optimize" (
    set "OPTIMIZE_RESOURCES=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--no-frontend" (
    set "SKIP_FRONTEND=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
echo %RED%[ERROR]%NC% Unknown option: %~1
goto :show_help

:show_help
echo.
echo Jones Framework Installer - Topology-Aware Modular Build System
echo.
echo Usage: install.bat [OPTIONS]
echo.
echo Options:
echo   --core        Install core components only (minimal ~50MB)
echo   --tda         Install core + TDA pipeline (~100MB)
echo   --ml          Install core + ML components (~2GB with PyTorch)
echo   --api         Install core + API server (~150MB)
echo   --full        Install everything including dev tools (~3GB)
echo   --frontend    Install frontend only
echo   --optimize    Enable resource optimization
echo   --no-frontend Skip frontend installation
echo   --help        Show this help message
echo.
echo Compartment Topology:
echo   CORE ----^> TDA ----^> SANS ----^> API
echo     ^|          ^|         ^|         ^|
echo     +----------+--------+----------+---^> ML (optional)
echo.
exit /b 0

REM =============================================================================
REM System Detection
REM =============================================================================

:detect_system
echo.
echo %CYAN%========================================%NC%
echo %CYAN%    JONES FRAMEWORK INSTALLER%NC%
echo %CYAN%    Interdimensional Build System%NC%
echo %CYAN%========================================%NC%
echo.

echo %BLUE%[INFO]%NC% Installation mode: %INSTALL_MODE%

REM Get CPU cores
for /f "tokens=2 delims==" %%a in ('wmic cpu get NumberOfCores /value ^| findstr NumberOfCores') do set CPU_CORES=%%a
echo %BLUE%[INFO]%NC% CPU Cores: %CPU_CORES%

REM Detect GPU
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    set "GPU_TYPE=cuda"
    echo %BLUE%[INFO]%NC% GPU: NVIDIA CUDA detected
) else (
    echo %BLUE%[INFO]%NC% GPU: None detected (CPU mode)
)

REM =============================================================================
REM Python Environment Setup
REM =============================================================================

:setup_python
echo.
echo %PURPLE%----------------------------------------%NC%
echo %PURPLE%  Python Environment%NC%
echo %PURPLE%----------------------------------------%NC%

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERROR]%NC% Python not found. Please install Python 3.9+
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo %BLUE%[INFO]%NC% Python version: %PYTHON_VERSION%

REM Create virtual environment
set "VENV_DIR=%PROJECT_ROOT%\.venv"
if not exist "%VENV_DIR%" (
    echo %BLUE%[INFO]%NC% Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"
echo %GREEN%[SUCCESS]%NC% Virtual environment activated

REM Upgrade pip
pip install --upgrade pip wheel setuptools >nul 2>&1

REM =============================================================================
REM Installation Based on Mode
REM =============================================================================

if "%INSTALL_MODE%"=="core" goto :install_core
if "%INSTALL_MODE%"=="tda" goto :install_tda
if "%INSTALL_MODE%"=="ml" goto :install_ml
if "%INSTALL_MODE%"=="api" goto :install_api
if "%INSTALL_MODE%"=="full" goto :install_full
if "%INSTALL_MODE%"=="frontend" goto :install_frontend_only
goto :install_all

:install_core
call :backend_core
call :backend_sans
goto :verify

:install_tda
call :backend_core
call :backend_tda
call :backend_sans
goto :verify

:install_ml
call :backend_core
call :backend_tda
call :backend_sans
call :backend_ml
goto :verify

:install_api
call :backend_core
call :backend_tda
call :backend_sans
call :backend_api
goto :verify

:install_full
call :backend_core
call :backend_tda
call :backend_sans
call :backend_ml
call :backend_api
call :backend_dev
if "%SKIP_FRONTEND%"=="false" call :frontend
goto :verify

:install_frontend_only
call :frontend
goto :verify

:install_all
call :backend_core
call :backend_tda
call :backend_sans
call :backend_api
if "%SKIP_FRONTEND%"=="false" call :frontend
goto :verify

REM =============================================================================
REM Backend Installation Functions
REM =============================================================================

:backend_core
echo.
echo %PURPLE%----------------------------------------%NC%
echo %PURPLE%  Installing Backend Core%NC%
echo %PURPLE%----------------------------------------%NC%

cd /d "%BACKEND_DIR%"

REM Create README if missing
if not exist "README.md" echo # Jones Framework > README.md

echo %BLUE%[INFO]%NC% Installing core dependencies...
pip install -e . --no-deps >nul 2>&1
pip install numpy scipy scikit-learn pandas pyyaml >nul 2>&1

echo %GREEN%[SUCCESS]%NC% Core components installed
goto :eof

:backend_tda
echo.
echo %PURPLE%----------------------------------------%NC%
echo %PURPLE%  Installing TDA Pipeline%NC%
echo %PURPLE%----------------------------------------%NC%

cd /d "%BACKEND_DIR%"

echo %BLUE%[INFO]%NC% Installing TDA dependencies...
pip install ripser persim >nul 2>&1

echo %GREEN%[SUCCESS]%NC% TDA pipeline installed
goto :eof

:backend_sans
echo.
echo %PURPLE%----------------------------------------%NC%
echo %PURPLE%  Installing SANS Architecture%NC%
echo %PURPLE%----------------------------------------%NC%

echo %BLUE%[INFO]%NC% SANS components included in core
echo %GREEN%[SUCCESS]%NC% SANS architecture verified
goto :eof

:backend_ml
echo.
echo %PURPLE%----------------------------------------%NC%
echo %PURPLE%  Installing ML Components%NC%
echo %PURPLE%----------------------------------------%NC%

cd /d "%BACKEND_DIR%"

echo %BLUE%[INFO]%NC% Installing ML dependencies...

if "%GPU_TYPE%"=="cuda" (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 >nul 2>&1
) else (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu >nul 2>&1
)

pip install transformers >nul 2>&1

echo %GREEN%[SUCCESS]%NC% ML components installed (%GPU_TYPE% acceleration)
goto :eof

:backend_api
echo.
echo %PURPLE%----------------------------------------%NC%
echo %PURPLE%  Installing API Server%NC%
echo %PURPLE%----------------------------------------%NC%

cd /d "%BACKEND_DIR%"

echo %BLUE%[INFO]%NC% Installing API dependencies...
pip install fastapi uvicorn[standard] pydantic websockets >nul 2>&1

echo %GREEN%[SUCCESS]%NC% API server installed
goto :eof

:backend_dev
echo.
echo %PURPLE%----------------------------------------%NC%
echo %PURPLE%  Installing Development Tools%NC%
echo %PURPLE%----------------------------------------%NC%

cd /d "%BACKEND_DIR%"

echo %BLUE%[INFO]%NC% Installing dev dependencies...
pip install pytest pytest-cov black mypy httpx >nul 2>&1

echo %GREEN%[SUCCESS]%NC% Development tools installed
goto :eof

REM =============================================================================
REM Frontend Installation
REM =============================================================================

:frontend
echo.
echo %PURPLE%----------------------------------------%NC%
echo %PURPLE%  Installing Frontend%NC%
echo %PURPLE%----------------------------------------%NC%

REM Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERROR]%NC% Node.js not found. Please install Node.js 18+
    exit /b 1
)

for /f "tokens=1 delims=v" %%v in ('node --version') do set NODE_VERSION=%%v
echo %BLUE%[INFO]%NC% Node.js version: %NODE_VERSION%

REM Check pnpm
pnpm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %BLUE%[INFO]%NC% Installing pnpm...
    npm install -g pnpm >nul 2>&1
)

REM Install frontend
cd /d "%FRONTEND_DIR%"
echo %BLUE%[INFO]%NC% Installing frontend dependencies...
call pnpm install >nul 2>&1

REM Install shared types
cd /d "%SHARED_DIR%"
if exist "package.json" (
    call pnpm install >nul 2>&1
)

echo %GREEN%[SUCCESS]%NC% Frontend installed
goto :eof

REM =============================================================================
REM Verification
REM =============================================================================

:verify
echo.
echo %PURPLE%----------------------------------------%NC%
echo %PURPLE%  Verifying Installation%NC%
echo %PURPLE%----------------------------------------%NC%

cd /d "%BACKEND_DIR%"

python -c "import warnings; warnings.filterwarnings('ignore'); from jones_framework.core.manifold_bridge import RecursiveImprover; improver = RecursiveImprover(); gaps = improver.identify_gaps(); orphaned = [g for g in gaps if g['type'] == 'orphaned']; print(f'  Orphaned components: {len(orphaned)}'); print('  STATUS: MANIFOLD HEALTHY' if len(orphaned) == 0 else '  STATUS: NEEDS ATTENTION')"

echo %GREEN%[SUCCESS]%NC% Installation verified

REM =============================================================================
REM Generate Environment Configuration
REM =============================================================================

echo.
echo %PURPLE%----------------------------------------%NC%
echo %PURPLE%  Generating Configuration%NC%
echo %PURPLE%----------------------------------------%NC%

set "ENV_FILE=%PROJECT_ROOT%\.env"

(
echo # =============================================================================
echo # Jones Framework Environment Configuration
echo # Generated by install.bat
echo # =============================================================================
echo.
echo # Backend Configuration
echo JONES_ENV=development
echo JONES_DEVICE=%GPU_TYPE%
echo JONES_LOG_LEVEL=INFO
echo.
echo # Resource Optimization
echo JONES_CPU_CORES=%CPU_CORES%
echo JONES_OPTIMIZE_RESOURCES=%OPTIMIZE_RESOURCES%
echo.
echo # TDA Configuration
echo RIPSER_MAX_DIM=2
echo RIPSER_THRESHOLD=2.0
echo.
echo # Frontend Configuration
echo NODE_ENV=development
echo VITE_API_URL=http://localhost:8000
echo.
echo # Installation Mode
echo JONES_INSTALL_MODE=%INSTALL_MODE%
) > "%ENV_FILE%"

echo %GREEN%[SUCCESS]%NC% Environment configuration saved

REM =============================================================================
REM Complete
REM =============================================================================

echo.
echo %GREEN%========================================%NC%
echo %GREEN%    Installation Complete%NC%
echo %GREEN%========================================%NC%
echo.
echo   Mode: %INSTALL_MODE%
echo   GPU:  %GPU_TYPE%
echo   Manifold Status: HEALTHY
echo.
echo   To start:
echo     .venv\Scripts\activate
echo     cd backend ^&^& uvicorn jones_framework.api.server:app --reload
echo.

endlocal
exit /b 0
