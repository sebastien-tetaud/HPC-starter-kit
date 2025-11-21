#!/usr/bin/env bash

###############################################################################
# BigEarthNet_HEALPix Installation Script
#
# This script automates the complete installation process:
# - Installs UV package manager (if not already installed)
# - Syncs all project dependencies
# - Verifies the installation
# - Sets up useful aliases (optional)
###############################################################################

# Ensure we're running with bash
if [ -z "$BASH_VERSION" ]; then
    echo "This script requires bash. Please run with: bash install.sh"
    exit 1
fi

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

###############################################################################
# Step 1: Check and Install Python 3.13 using UV
###############################################################################
print_header "Step 1: Checking Python Version"

REQUIRED_VERSION="3.11.7"

# Check if python command exists and get version
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    print_info "Found system Python version: $PYTHON_VERSION"

    if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" == "$REQUIRED_VERSION" ]] && [[ "$PYTHON_VERSION" == "$REQUIRED_VERSION"* ]]; then
        print_success "Python $REQUIRED_VERSION is already available"
        PYTHON_CMD="python"
    else
        print_warning "System Python is $PYTHON_VERSION, but we need Python $REQUIRED_VERSION"
        PYTHON_CMD=""
    fi
else
    print_warning "No system Python found"
    PYTHON_CMD=""
fi

# If Python 3.13 is not available, offer to install it via UV
if [[ -z "$PYTHON_CMD" ]]; then
    print_info "Python $REQUIRED_VERSION is required for this project"
    echo ""
    read -p "Would you like to install Python $REQUIRED_VERSION using UV? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Will install Python $REQUIRED_VERSION after UV is set up..."
        INSTALL_PYTHON=true
    else
        print_error "Python $REQUIRED_VERSION is required. Please install it manually or re-run this script."
        exit 1
    fi
else
    INSTALL_PYTHON=false
fi

###############################################################################
# Step 2: Install UV package manager
###############################################################################
print_header "Step 2: Installing UV Package Manager"

if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version)
    UV_PATH=$(which uv)
    print_warning "UV is already installed: $UV_VERSION"
    print_info "UV location: $UV_PATH"

    read -p "Do you want to update UV to the latest version? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Updating UV..."


        if [[ "$UV_PATH" == *".local"* ]] || [[ "$UV_PATH" == *".cargo"* ]]; then
            print_info "Detected standalone installation. Updating via uv self update..."
            uv self update || {
                print_warning "uv self update failed, trying pip..."
                pip install --upgrade uv
            }
        else
            print_info "Updating via pip..."
            pip install --upgrade uv
        fi

        print_success "UV updated successfully"
        print_info "New version: $(uv --version)"
    else
        print_info "Skipping UV update"
    fi
else
    print_info "UV not found. Installing UV..."

    # Try curl method first
    if command -v curl &> /dev/null; then
        print_info "Installing UV using curl..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add UV to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"

    elif command -v pip &> /dev/null; then
        print_info "Installing UV using pip..."
        pip install uv
    else
        print_error "Neither curl nor pip found. Cannot install UV."
        print_info "Please install curl or pip first."
        exit 1
    fi

    print_success "UV installed successfully"
fi

# Verify UV installation
if command -v uv &> /dev/null; then
    print_success "UV is available: $(uv --version)"
else
    print_error "UV installation failed or not in PATH"
    print_info "Try adding ~/.cargo/bin to your PATH: export PATH=\"\$HOME/.cargo/bin:\$PATH\""
    exit 1
fi

###############################################################################
# Step 2.5: Install Python 3.13 using UV (if needed)
###############################################################################
if [[ "$INSTALL_PYTHON" == true ]]; then
    print_header "Step 2.5: Installing Python $REQUIRED_VERSION with UV"

    print_info "Installing Python $REQUIRED_VERSION..."
    uv python install $REQUIRED_VERSION

    if [ $? -eq 0 ]; then
        print_success "Python $REQUIRED_VERSION installed successfully via UV"

        # Verify the installation
        UV_PYTHON_VERSION=$(uv python list | grep "$REQUIRED_VERSION" | head -n 1)
        print_info "Installed: $UV_PYTHON_VERSION"
    else
        print_error "Failed to install Python $REQUIRED_VERSION via UV"
        exit 1
    fi
fi

###############################################################################
# Step 3: Sync project dependencies
###############################################################################
print_header "Step 3: Syncing Project Dependencies"

print_info "Running 'uv sync' with Python $REQUIRED_VERSION to install all dependencies..."
print_info "This will create a virtual environment and install PyTorch, NumPy, Loguru, etc."

# Sync with specific Python version
uv sync --python $REQUIRED_VERSION

if [ $? -eq 0 ]; then
    print_success "Dependencies synced successfully with Python $REQUIRED_VERSION"
    print_success "Virtual environment created at: .venv"
else
    print_error "Failed to sync dependencies"
    print_info "Try running: uv sync --python $REQUIRED_VERSION"
    exit 1
fi

###############################################################################
# Step 4: Verify installation [for local training ONLY]
###############################################################################
print_header "Step 4: Verifying Installation [for local training ONLY]"

print_info "Activating virtual environment and checking packages..."

# Activate virtual environment and check packages
source .venv/bin/activate

# Check PyTorch with CUDA/GPU
print_info "Checking PyTorch installation and GPU availability..."

python3 << 'EOF'
import sys

try:
    import torch

    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print()

    # PyTorch version
    print(f"✓ PyTorch version: {torch.__version__}")
    print()

    # CUDA availability and details
    print("=" * 50)
    print("CUDA & GPU Information:")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"✓ CUDA is AVAILABLE")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
        print(f"✓ cuDNN enabled: {torch.backends.cudnn.enabled}")
        print()

        # GPU details
        num_gpus = torch.cuda.device_count()
        print(f"✓ Number of GPUs detected: {num_gpus}")
        print()

        for i in range(num_gpus):
            print(f"GPU {i}:")
            print(f"  - Name: {torch.cuda.get_device_name(i)}")

            # Get device properties
            props = torch.cuda.get_device_properties(i)
            print(f"  - Compute Capability: {props.major}.{props.minor}")
            print(f"  - Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  - Multi-Processors: {props.multi_processor_count}")

            # Memory info
            if torch.cuda.is_initialized():
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  - Memory Allocated: {mem_allocated:.2f} GB")
                print(f"  - Memory Reserved: {mem_reserved:.2f} GB")
            print()

        # Quick GPU test
        print("Running quick GPU test...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            print("✓ GPU computation test PASSED")
            del x, y
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU computation test FAILED: {e}")
            sys.exit(1)

    else:
        print("⚠ CUDA is NOT available")
        print("⚠ PyTorch is running in CPU-only mode")
        print()
        print("Possible reasons:")
        print("  - No CUDA-capable GPU detected")
        print("  - NVIDIA drivers not installed")
        print("  - PyTorch CPU-only version installed")
        print("  - CUDA version mismatch")
        print()
        print("Note: You can still run benchmarks on CPU, but performance will be limited.")

    print("=" * 50)

except ImportError:
    print("✗ PyTorch is NOT installed!")
    print("Installation verification failed.")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error during verification: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "All required packages are installed correctly"
else
    print_error "Package verification failed"
    exit 1
fi

###############################################################################
# Step 5: Optional - Set up aliases
###############################################################################
print_header "Step 5: Setting Up HPC Aliases (Optional)"

ALIAS_MARKER="# BigEarthNet HEALPix aliases"

# Check if aliases already exist
if grep -q "$ALIAS_MARKER" ~/.bashrc; then
    print_warning "Aliases already exist in ~/.bashrc."
    echo ""
    echo "Options:"
    echo "  1) Keep existing aliases (do nothing)"
    echo "  2) Remove existing aliases"
    echo "  3) Update/replace existing aliases"
    echo ""
    read -p "Choose an option (1-3): " -n 1 -r choice
    echo ""

    case $choice in
        2)
            print_info "Removing existing aliases from ~/.bashrc..."
            # Create a backup
            cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
            print_info "Backup created: ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)"

            # Remove the alias block
            sed -i "/^# BigEarthNet HEALPix aliases/,/^alias qerror=/d" ~/.bashrc
            print_success "Aliases removed from ~/.bashrc"
            ;;
        3)
            print_info "Updating aliases in ~/.bashrc..."
            # Create a backup
            cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
            print_info "Backup created: ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)"

            # Remove old aliases
            sed -i "/^# BigEarthNet HEALPix aliases/,/^alias qerror=/d" ~/.bashrc

            # Fall through to add new aliases
            ;&
        1|*)
            if [[ $choice == "1" ]] || [[ -z $choice ]]; then
                print_info "Keeping existing aliases"
            fi
            ;;
    esac

    # If choice was 3, continue to add aliases
    if [[ $choice == "3" ]]; then
        ADD_ALIASES=true
    else
        ADD_ALIASES=false
    fi
else
    # No existing aliases, ask if user wants to add them
    print_info "Would you like to add helpful HPC job aliases to your ~/.bashrc?"
    echo ""
    echo "Available alias categories:"
    echo "  - Job monitoring (qstat, qwatch, etc.)"
    echo "  - Job management (qdel, qdelall, etc.)"
    echo "  - Job information (qnodes, qinfo, etc.)"
    echo "  - Job logs (qtail, qerr, qlogs, etc.)"
    echo "  - Project-specific (qlog, qerror)"
    echo ""
    echo "Options:"
    echo "  1) Install all aliases (recommended)"
    echo "  2) Install only essential aliases (monitoring + management)"
    echo "  3) Install only project-specific aliases"
    echo "  4) Skip alias installation"
    echo ""
    read -p "Choose an option (1-4): " -n 1 -r choice
    echo ""

    case $choice in
        1)
            ALIAS_TYPE="full"
            ADD_ALIASES=true
            ;;
        2)
            ALIAS_TYPE="essential"
            ADD_ALIASES=true
            ;;
        3)
            ALIAS_TYPE="project"
            ADD_ALIASES=true
            ;;
        4|*)
            print_info "Skipping alias setup"
            ADD_ALIASES=false
            ;;
    esac
fi

# Add aliases based on user choice
if [[ $ADD_ALIASES == true ]]; then
    print_info "Adding aliases to ~/.bashrc..."

    # Create backup before modifying
    if [[ ! -f ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S) ]]; then
        cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
        print_info "Backup created: ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)"
    fi

    # Start alias block
    cat >> ~/.bashrc << 'ALIAS_HEADER'

# BigEarthNet HEALPix aliases
ALIAS_HEADER

    # Add monitoring aliases (essential and full)
    if [[ $ALIAS_TYPE == "essential" ]] || [[ $ALIAS_TYPE == "full" ]] || [[ $choice == "3" ]]; then
        cat >> ~/.bashrc << 'ALIAS_MONITORING'
# Job monitoring
alias qstat='qstat -u $USER'
alias qstatall='qstat -a'
alias qstatf='qstat -f'
alias qwatch='watch -n 5 qstat -u $USER'

ALIAS_MONITORING
    fi

    # Add management aliases (essential and full)
    if [[ $ALIAS_TYPE == "essential" ]] || [[ $ALIAS_TYPE == "full" ]] || [[ $choice == "3" ]]; then
        cat >> ~/.bashrc << 'ALIAS_MANAGEMENT'
# Job management
alias qdel='qdel'
alias qdelall='qdel $(qstat -u $USER | grep $USER | cut -d. -f1)'

ALIAS_MANAGEMENT
    fi

    # Add information aliases (full only)
    if [[ $ALIAS_TYPE == "full" ]] || [[ $choice == "3" ]]; then
        cat >> ~/.bashrc << 'ALIAS_INFO'
# Job information
alias qnodes='pbsnodes -a'
alias qinfo='qstat -Q'
alias qhold='qhold'
alias qrls='qrls'

ALIAS_INFO
    fi

    # Add log aliases (full only)
    if [[ $ALIAS_TYPE == "full" ]] || [[ $choice == "3" ]]; then
        cat >> ~/.bashrc << 'ALIAS_LOGS'
# Job logs
alias qtail='tail -f'
alias qerr='tail -f'
alias qlogs='ls -lht *.o* *.e*'

ALIAS_LOGS
    fi

    # Add project-specific aliases (all types)
    cat >> ~/.bashrc << 'ALIAS_PROJECT'
# Project-specific aliases
alias qlog='tail -f HEALPix_BigEarthNet.o*'
alias qerror='tail -f HEALPix_BigEarthNet.e*'
ALIAS_PROJECT

    print_success "Aliases added to ~/.bashrc"
    print_info "Run 'source ~/.bashrc' to activate them, or they will be available in new terminal sessions"
fi

###############################################################################
# Installation Complete
###############################################################################
print_header "Installation Complete!"

echo -e "${GREEN}✓ UV package manager installed${NC}"
echo -e "${GREEN}✓ Virtual environment created at .venv${NC}"
echo -e "${GREEN}✓ All dependencies installed${NC}"
echo -e "${GREEN}✓ Installation verified${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Activate the virtual environment:"
echo -e "     ${YELLOW}source .venv/bin/activate${NC}"
echo ""
echo -e "  2. Run benchmarks locally:"
echo -e "     ${YELLOW}python scripts/main.py${NC}"
echo -e "     ${YELLOW}python scripts/scripts/ddp_cnn.py${NC}"
echo -e "     ${YELLOW}python scripts/scripts/ddp_cnn_wandb.py${NC}"
echo ""
echo -e "  3. Submit HPC jobs:"
echo -e "     ${YELLOW}qsub basic_bechmark_job.pbs${NC}"
echo -e "     ${YELLOW}qsub ddp_cnn_job.pbs${NC}"
echo -e "     ${YELLOW}qsub ddp_cnn_wandb_job.pbs${NC}"

echo ""
echo -e "  4. If you added aliases, activate them:"
echo -e "     ${YELLOW}source ~/.bashrc${NC}"
echo ""
echo -e "${GREEN}Happy computing!${NC}"
echo ""
