#!/bin/bash

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

detect_os() {
    case "$(uname -s)" in
        Linux*)     
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                echo "$ID"
            else
                echo "linux"
            fi
            ;;
        Darwin*)    echo "macos" ;;
        CYGWIN*)   echo "windows" ;;
        MINGW*)    echo "windows" ;;
        *)         echo "unknown" ;;
    esac
}

install_package_manager() {
    local os=$1
    case $os in
        ubuntu|debian)
            if ! command_exists apt-get; then
                echo "❌ apt-get not found. Please install it manually."
                exit 1
            fi
            sudo apt-get update
            ;;
        fedora)
            if ! command_exists dnf; then
                echo "❌ dnf not found. Please install it manually."
                exit 1
            fi
            sudo dnf check-update
            ;;
        macos)
            if ! command_exists brew; then
                echo "🍺 Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            ;;
    esac
}

install_python() {
    local os=$1
    if ! command_exists python3; then
        echo "🐍 Installing Python 3..."
        case $os in
            ubuntu|debian)
                sudo apt-get install -y python3 python3-pip python3-venv
                ;;
            fedora)
                sudo dnf install -y python3 python3-pip python3-virtualenv
                ;;
            macos)
                brew install python
                ;;
            *)
                echo "❌ Unsupported OS for automatic Python installation. Please install Python 3 manually."
                exit 1
                ;;
        esac
    fi
}

install_node() {
    local os=$1
    if ! command_exists node || ! command_exists npm; then
        echo "📦 Installing Node.js and npm..."
        case $os in
            ubuntu|debian)
                curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
                sudo apt-get install -y nodejs
                ;;
            fedora)
                sudo dnf install -y nodejs npm
                ;;
            macos)
                brew install node
                ;;
            *)
                echo "❌ Unsupported OS for automatic Node.js installation. Please install Node.js manually."
                exit 1
                ;;
        esac
    fi
}

echo "🐱 Setting up Catology application..."

OS=$(detect_os)
echo "🖥️ Detected OS: $OS"
install_package_manager $OS
install_python $OS
install_node $OS

echo "✅ Verifying installations..."
python3 --version
node --version
npm --version

echo "🐍 Setting up Python backend..."
cd catology-backend || exit 1
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing Python dependencies..."
python3 installdeps.py

echo "⚛️ Setting up React frontend..."
cd ../catology-frontend || exit 1

echo "📦 Installing Node.js dependencies..."
npm install
npm install -D typescript @types/node @types/react @types/react-dom @types/jest @testing-library/react @testing-library/jest-dom @testing-library/user-event

cleanup() {
    echo "🧹 Cleaning up..."
    
    if [ ! -z "$frontend_pid" ] && kill -0 $frontend_pid 2>/dev/null; then
        echo "Stopping frontend server..."
        kill -SIGINT $frontend_pid
        wait $frontend_pid 2>/dev/null
    fi

    if [ ! -z "$backend_pid" ] && kill -0 $backend_pid 2>/dev/null; then
        echo "Stopping backend server..."
        kill -SIGINT $backend_pid
        sleep 1
        if kill -0 $backend_pid 2>/dev/null; then
            kill -9 $backend_pid 2>/dev/null
        fi
        wait $backend_pid 2>/dev/null
    fi
    pkill -f "python3 catology.py" 2>/dev/null
    sleep 1
    echo "✨ All processes stopped"
    exit 0
}

trap cleanup EXIT
trap cleanup SIGINT
trap cleanup SIGTERM

echo "🚀 Starting Catology..."
cd ../catology-backend || exit 1
FLASK_ENV=development PYTHONUNBUFFERED=1 python3 catology.py &
backend_pid=$!
sleep 2

cd ../catology-frontend || exit 1
npm start &
frontend_pid=$!

echo "✨ Setup complete! Frontend will be available at http://localhost:3000"
echo "✨ Backend will be available at http://127.0.0.1:5000"
echo "⌨️ Press Ctrl+C to stop both servers"

wait $backend_pid $frontend_pid