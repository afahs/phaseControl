# Raspberry Pi handoff

Last verified: October 12, 2026.

## Owner's intent and preferences

Prepare this Pi for a new project whose requirements will come later. Use SSH keys for
routine access and Git for version control and deployment, so updates do not require
interactive SSH sessions. Keep password login enabled on the Pi for fallback access.

## Local workspace

- Directory: `C:\Users\Adam\Documents\ChatGPT\ClockSyncing`
- Branch: `main`, tracking `pi/main`.
- Commit identity, configured only for this repository: `Adam <adam.fahs@berkeley.edu>`.
- Remote `pi`: `project-pi:/home/rrrodriguez13/projects/new-project`.
- No hosted `origin` has been configured.

## Access

- Address: `192.168.0.145`, SSH port 22.
- Login: `rrrodriguez13`.
- SSH alias on this Windows computer: `project-pi`.
- Pi hostname: `pi3`.
- OS observed: Debian 12 (Bookworm); Git 2.39.2.
- SSH config: `C:\Users\Adam\.ssh\config`.
- Private key: `C:\Users\Adam\.ssh\id_ed25519_project_pi`.
- Public key: the same path with `.pub` appended.
- Key type: Ed25519; no passphrase, for unattended updates.
- Key fingerprint: `SHA256:QXP1dfWfgDMrBTN9LvIG8Oooalz8Zti7gK4OODGXR4U`.
- Public key is installed in the Pi account's `~/.ssh/authorized_keys`.
- Host key is already recorded in `C:\Users\Adam\.ssh\known_hosts`.
- Alias requires public-key authentication and strict host-key checking.
- Pi server allows both public-key and password authentication.
- `sudo -n true` succeeded for this account during setup.

Connect or run a one-off command:

```powershell
ssh project-pi
ssh -o BatchMode=yes -o ConnectTimeout=10 project-pi 'hostname; id'
```

Explicit password fallback, if the owner needs it:

```powershell
ssh -o PreferredAuthentications=password -o PasswordAuthentication=yes -o PubkeyAuthentication=no project-pi
```

The password and private key contents are not stored in project files. Keys and SSH config
exist on this Windows computer, outside this repository; cloning the repository onto
another computer does not install access credentials.

If a Codex sandbox blocks LAN connections, retry the authorized SSH/Git operation with
the tool's network escalation. Do not interpret a sandbox-level denial as a Pi login failure.
If the host key changes, verify the Pi's identity before replacing the saved host key.

## Git deployment

The live checkout is `/home/rrrodriguez13/projects/new-project` on `main`.
It uses Git's `receive.denyCurrentBranch=updateInstead` to update tracked files when pushed.
`receive.denyNonFastForwards=true` and `receive.denyDeletes=true` protect history.
`setup-pi.sh` records the provisioning steps and refuses to initialize an existing
non-repository directory. Git was already installed; no OS upgrade was performed.

From the local workspace:

```powershell
git status
git add <files>
git commit -m "Describe the change"
git push
```

`git push pi main` is the explicit equivalent. Keep tracked files clean on the Pi, since
Git rejects pushes if the checkout has local modifications. Keep runtime data, logs,
and secrets outside tracked source files. Roll back by reverting locally and pushing
the resulting commit; do not force-push.

Repository-local `core.sshCommand`:

```text
C:/Users/Adam/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/usr/bin/ssh.exe -F C:/Users/Adam/.ssh/config -o BatchMode=yes -o ConnectTimeout=10
```

The Windows system SSH transport initially disconnected after transferring a commit.
Git was then configured to use its bundled SSH client. A subsequent push of a new commit
completed successfully; the Pi's HEAD matched the local commit and its checkout was clean.
Connections sometimes took longer than expected during setup. If a push reports a
disconnect, check both HEADs before deciding whether it failed to deploy.
If the bundled runtime path changes, update `core.sshCommand` to the installed Git SSH client.

Verify deployment:

```powershell
git rev-parse HEAD
ssh -o BatchMode=yes -o ConnectTimeout=10 project-pi 'git -C ~/projects/new-project rev-parse HEAD; git -C ~/projects/new-project status --porcelain'
```

The hashes should match and the remote status should be empty.

## Work remaining once project details arrive

- Choose the application, dependencies, and runtime.
- Add service startup, automatic restarts, and health checks as appropriate.
- Git currently deploys source files only; it does not restart applications or install dependencies.
- No scheduled deployment, CI workflow, or hosted Git backup has been configured.
- Router DHCP reservation for `192.168.0.145` was recommended but has not been verified or configured.
- The placeholder Pi directory `new-project` can be renamed when the project is defined;
  update the local `pi` remote and provisioning script if doing so.
