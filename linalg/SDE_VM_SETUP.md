# Step-by-Step Guide: Set up x86_64 Linux VM with Intel SDE for ACE Testing

## Step 1: Create x86_64 Linux VM in UTM

1. **Open UTM** (already launched)
2. **Create new VM**:
   - Click "Create a New Virtual Machine"
   - Select "Virtualize" 
   - Choose "Linux" as operating system
   - Select "Ubuntu 22.04 LTS" (or similar x86_64 Linux)
3. **Configure VM settings**:
   - Memory: 4GB+ RAM (recommend 8GB for better performance)
   - CPU: 4+ cores with AVX-512 support enabled
   - Storage: 20GB+ disk space
   - Network: Enable network access

## Step 2: Install Ubuntu and Basic Tools

1. **Boot the VM** and follow Ubuntu installation prompts
2. **Update system**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
3. **Install build tools**:
   ```bash
   sudo apt install -y build-essential gcc nasm binutils git wget
   ```

## Step 3: Download and Install Intel SDE

1. **Download Intel SDE 10.13.1** (latest with ACE support):
   ```bash
   cd ~
   wget https://downloadmirror.intel.com/813591/sde-external-10.13.1-2026-07-28-lin.tar.xz
   ```

2. **Extract SDE**:
   ```bash
   tar xf sde-external-10.13.1-2026-07-28-lin.tar.xz
   cd sde-external-10.13.1-2026-07-28-lin
   ```

3. **Add to PATH** (temporary for session):
   ```bash
   export PATH=$PATH:$PWD
   ```

4. **Verify installation**:
   ```bash
   sde --version
   ```

## Step 4: Test ACE Support in SDE

1. **Test basic SDE operation**:
   ```bash
   sde -future-ag -- echo "ACE support test"
   ```

2. **Verify ACE instructions are available**:
   ```bash
   sde -future-ag -- /bin/true
   ```

## Step 5: Clone tract Repository in VM

```bash
cd ~
git clone https://github.com/sonos/tract.git
cd tract
git checkout czoli1976/ace-upstream
```

## Step 6: Run ACE Tests with SDE

Once SDE is working, you can run the differential tests:

```bash
cd ~/tract
cargo test -p tract-linalg --test ace_sde_tests
```

The tests will use the SDE infrastructure we created to compare the software model against SDE's ACE emulation.

## Alternative: Quick Test Without Full Setup

If you want to quickly test SDE without full VM setup, you can:

1. Create a minimal x86_64 Linux VM with just SDE
2. Write a simple ACE test program in assembly
3. Run it under SDE to verify ACE emulation works

## Troubleshooting

**If SDE download fails**: Try alternative mirror or check Intel's download page
**If VM performance is slow**: Increase RAM/CPU allocation in UTM settings
**If ACE instructions not recognized**: Ensure you're using SDE 10.13.1+ with `-future-ag` flag

## Next Steps After VM Setup

Once the VM is set up and SDE is working, let me know and I can help you:
1. Create simple ACE test programs
2. Run differential tests against the software model
3. Validate bit-exactness between model and SDE
4. Document any discrepancies found