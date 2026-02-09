"""Interactive installation wizard engine."""

from enum import Enum, auto
from typing import Optional, Callable
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from jones_framework.cli.constants import InstallMode, INSTALL_MODES
from jones_framework.cli.wizard.screens import WizardScreens
from jones_framework.cli.ui.output import print_info, print_success, print_error, print_warning


class WizardState(Enum):
    """States in the wizard flow."""
    WELCOME = auto()
    SYSTEM_CHECK = auto()
    MODE_SELECT = auto()
    CONFIRM = auto()
    INSTALL = auto()
    COMPLETE = auto()
    CANCELLED = auto()
    ERROR = auto()


class WizardEngine:
    """Interactive installation wizard."""

    def __init__(self):
        self.console = Console()
        self.screens = WizardScreens(self.console)
        self.state = WizardState.WELCOME
        self.selected_mode: InstallMode = InstallMode.STANDARD
        self.check_results: list = []
        self.error_message: Optional[str] = None

    def run(self) -> bool:
        """Run the wizard and return True if installation succeeded."""
        try:
            while self.state not in (WizardState.COMPLETE, WizardState.CANCELLED, WizardState.ERROR):
                self._process_state()

            return self.state == WizardState.COMPLETE

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Installation cancelled[/yellow]")
            return False

    def _process_state(self) -> None:
        """Process the current wizard state."""
        handlers = {
            WizardState.WELCOME: self._handle_welcome,
            WizardState.SYSTEM_CHECK: self._handle_system_check,
            WizardState.MODE_SELECT: self._handle_mode_select,
            WizardState.CONFIRM: self._handle_confirm,
            WizardState.INSTALL: self._handle_install,
        }

        handler = handlers.get(self.state)
        if handler:
            handler()

    def _handle_welcome(self) -> None:
        """Show welcome screen and proceed."""
        self.screens.show_welcome()

        choice = self.console.input("\nPress [Enter] to continue or [q] to quit: ").strip().lower()

        if choice == "q":
            self.state = WizardState.CANCELLED
        else:
            self.state = WizardState.SYSTEM_CHECK

    def _handle_system_check(self) -> None:
        """Run and display system checks."""
        from jones_framework.cli.services.dependency_checker import DependencyChecker

        self.screens.show_step(1, 4, "Checking your computer")

        checker = DependencyChecker()
        self.check_results = checker.get_full_report()

        self.screens.show_check_results(self.check_results)

        # Check for blocking issues
        blocking = [r for r in self.check_results if not r.passed and r.name == "Python version"]

        if blocking:
            self.console.print("\n[red]Cannot continue due to the issues above.[/red]")
            self.console.print("[dim]Please fix these issues and try again.[/dim]")
            self.state = WizardState.ERROR
            return

        # Check for warnings
        warnings = [r for r in self.check_results if not r.passed]
        if warnings:
            self.console.print(f"\n[yellow]{len(warnings)} issue(s) found, but we can continue.[/yellow]")

        choice = self.console.input("\nPress [Enter] to continue or [q] to quit: ").strip().lower()

        if choice == "q":
            self.state = WizardState.CANCELLED
        else:
            self.state = WizardState.MODE_SELECT

    def _handle_mode_select(self) -> None:
        """Let user select installation mode."""
        self.screens.show_step(2, 4, "Choose installation mode")

        self.selected_mode = self.screens.show_mode_selection()

        self.state = WizardState.CONFIRM

    def _handle_confirm(self) -> None:
        """Confirm installation settings."""
        self.screens.show_step(3, 4, "Confirm installation")

        mode_info = INSTALL_MODES[self.selected_mode]

        self.console.print(f"\n[bold]Ready to install:[/bold]")
        self.console.print(f"  Mode: [cyan]{mode_info.name}[/cyan]")
        self.console.print(f"  Size: [dim]{mode_info.size}[/dim]")
        self.console.print(f"  Features:")
        for feature in mode_info.features:
            self.console.print(f"    - {feature}")

        self.console.print()
        choice = self.console.input("Continue? [Y/n/b]: ").strip().lower()

        if choice == "n":
            self.state = WizardState.CANCELLED
        elif choice == "b":
            self.state = WizardState.MODE_SELECT
        else:
            self.state = WizardState.INSTALL

    def _handle_install(self) -> None:
        """Run the installation."""
        self.screens.show_step(4, 4, "Installing")

        from jones_framework.cli.services.install_manager import InstallManager

        def progress_callback(message: str, percent: int) -> None:
            pass  # Progress is shown via rich progress bar

        manager = InstallManager(progress_callback=progress_callback)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Installing...", total=100)

            # Create venv (10%)
            progress.update(task, description="Creating virtual environment...")
            if not manager.create_virtual_env():
                self.error_message = "Failed to create virtual environment"
                self.state = WizardState.ERROR
                return
            progress.update(task, advance=10)

            # Install Python deps (40%)
            progress.update(task, description="Installing Python packages...")
            if not manager.install_python_deps(self.selected_mode):
                self.error_message = "Failed to install Python dependencies"
                self.state = WizardState.ERROR
                return
            progress.update(task, advance=40)

            # Install Node deps if needed (40%)
            if self.selected_mode in (InstallMode.STANDARD, InstallMode.FULL):
                progress.update(task, description="Installing frontend packages...")
                if not manager.install_node_deps():
                    print_warning("Frontend dependencies could not be installed")
                    # Don't fail, just warn
            progress.update(task, advance=40)

            # Done
            progress.update(task, description="Complete!", advance=10)

        self.state = WizardState.COMPLETE
        self._show_completion()

    def _show_completion(self) -> None:
        """Show completion screen."""
        self.screens.show_completion()

        self.console.print("\n[bold green]Installation complete![/bold green]")
        self.console.print()
        self.console.print("What's next:")
        self.console.print("  [cyan]jones start[/cyan]    - Start the framework")
        self.console.print("  [cyan]jones status[/cyan]   - Check if everything is running")
        self.console.print("  [cyan]jones doctor[/cyan]   - If something goes wrong")
        self.console.print()


def run_wizard() -> bool:
    """
    Entry point function for the wizard.

    This is used by the `jones-wizard` console script entry point.
    """
    from jones_framework.cli.ui.banner import show_banner
    show_banner()
    engine = WizardEngine()
    return engine.run()
