# Free Cloud Setup Guide for XM360 Auto Trader

## Option 1: Oracle Cloud (RECOMMENDED - Always Free Forever)

### Step 1: Create Oracle Cloud Account
1. Go to: https://www.oracle.com/cloud/free/
2. Click "Start for free"
3. Fill in details (credit card for verification only - NOT charged)
4. Select your home region (choose nearest to you)

### Step 2: Create a Free VM
1. Login to Oracle Cloud Console
2. Go to: Compute → Instances → Create Instance
3. Configure:
   - **Name:** mt5-auto-trader
   - **Image:** Ubuntu 22.04 (or Windows Server if available in free tier)
   - **Shape:** VM.Standard.E2.1.Micro (Always Free)
   - **RAM:** 1 GB (enough for MT5 + bot)
4. Download the SSH key
5. Click "Create"

### Step 3: Connect to Your VM
```bash
# From your PC terminal:
ssh -i <your-key.pem> ubuntu@<your-vm-ip>
```

### Step 4: Install Required Software (Ubuntu)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Wine (to run MT5 on Linux)
sudo dpkg --add-architecture i386
sudo apt install -y wine64 wine32 winetricks

# Install Python
sudo apt install -y python3 python3-pip python3-venv

# Install display server (for MT5)
sudo apt install -y xvfb x11vnc

# Create virtual display
Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99
```

### Step 5: Install MetaTrader 5
```bash
# Download MT5
cd ~
wget https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe

# Install MT5 with Wine
wine mt5setup.exe

# MT5 will be installed in: ~/.wine/drive_c/Program Files/MetaTrader 5/
```

### Step 6: Upload Your Auto Trader Code
```bash
# On your local PC, use SCP to upload:
scp -i <your-key.pem> -r xm360_auto_trader ubuntu@<your-vm-ip>:~/

# Or use Git:
# On the VM:
git clone <your-repo-url>
```

### Step 7: Install Python Dependencies
```bash
cd ~/xm360_auto_trader
python3 -m venv venv
source venv/bin/activate
pip install MetaTrader5 python-telegram-bot requests
```

### Step 8: Configure MT5 Login
1. Start MT5: `wine ~/.wine/drive_c/Program\ Files/MetaTrader\ 5/terminal64.exe &`
2. Login with your XM360 credentials:
   - Account: 315982803
   - Password: Gadhiya@098
   - Server: XMGlobal-MT5 7
3. Enable Algo Trading in MT5 settings

### Step 9: Create Startup Script
```bash
cat > ~/start_trader.sh << 'EOF'
#!/bin/bash
export DISPLAY=:99

# Start virtual display
Xvfb :99 -screen 0 1024x768x16 &
sleep 2

# Start MT5
wine ~/.wine/drive_c/Program\ Files/MetaTrader\ 5/terminal64.exe &
sleep 10

# Start auto trader
cd ~/xm360_auto_trader
source venv/bin/activate
python auto_trader.py
EOF

chmod +x ~/start_trader.sh
```

### Step 10: Run on Startup (systemd)
```bash
sudo cat > /etc/systemd/system/auto-trader.service << 'EOF'
[Unit]
Description=XM360 Auto Trader
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/home/ubuntu/start_trader.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable auto-trader
sudo systemctl start auto-trader
```

---

## Option 2: AWS Free Tier (12 months free)

### Step 1: Create AWS Account
1. Go to: https://aws.amazon.com/free/
2. Create account (credit card needed for verification)
3. Select "Free Tier" eligible services only

### Step 2: Launch EC2 Instance
1. Go to EC2 → Launch Instance
2. Choose: **Windows Server 2022 Base** (easier for MT5)
3. Instance type: **t2.micro** (Free tier eligible)
4. Storage: 30 GB (free tier limit)
5. Create/select key pair
6. Launch!

### Step 3: Connect via RDP
1. Get Windows password from EC2 console
2. Use Remote Desktop to connect
3. Install MT5 and Python directly (no Wine needed!)

---

## Option 3: Google Cloud (Free $300 credits)

### Step 1: Create Google Cloud Account
1. Go to: https://cloud.google.com/free
2. Sign up (get $300 free credit for 90 days)

### Step 2: Create VM
1. Go to Compute Engine → VM Instances
2. Create instance:
   - **Machine type:** e2-micro (always free) or e2-small (use credits)
   - **Boot disk:** Windows Server 2022
   - **Region:** Any (some have more free resources)

---

## Quick Comparison

| Provider | Free Duration | RAM | Best For |
|----------|--------------|-----|----------|
| Oracle Cloud | **Forever** | 1GB | Long-term use |
| AWS | 12 months | 1GB | Easy Windows setup |
| Google Cloud | 90 days ($300) | Varies | Testing |

---

## Alternative: Use Your Own PC (Simplest!)

If you have a PC that can stay on:

### Step 1: Install MT5 Desktop
1. Download from: https://www.xm.com/mt5
2. Install and login with your credentials

### Step 2: Run the Bot
```cmd
cd xm360_auto_trader
python auto_trader.py
```

### Step 3: Keep PC Running
- Disable sleep mode
- Use Task Scheduler to auto-start

---

## Need Help?

Once you choose an option, I can help you with:
1. Step-by-step setup commands
2. Troubleshooting connection issues
3. Configuring auto-start
4. Testing the live connection

**Which option would you like to proceed with?**
- A) Oracle Cloud (Free forever)
- B) AWS Free Tier (Free 12 months)
- C) Google Cloud ($300 free credits)
- D) Use my own PC
