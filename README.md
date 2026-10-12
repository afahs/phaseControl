# Raspberry Pi project foundation

Target: `rrrodriguez13@192.168.0.145`, SSH alias `project-pi`.
Host: `pi3`, Debian 12 (Bookworm), Git 2.39.2.
The application and runtime will be chosen later.

Future conversations should read [PI_HANDOFF.md](PI_HANDOFF.md) for connection details,
setup history, owner preferences, and remaining work. `AGENTS.md` directs agents to it.

## SSH

The dedicated private key stays on this Windows computer at
`C:/Users/Adam/.ssh/id_ed25519_project_pi`; never commit or copy it to the Pi.
It has no passphrase for unattended updates and is protected by Windows file permissions.
The SSH alias uses public-key authentication only and the existing known host record.
The public key is installed and key-only login has been verified. Connect with `ssh project-pi`.
Server password authentication remains enabled at the owner's request, for fallback access.
This computer's SSH alias uses keys only. To explicitly use a password from this computer:

```powershell
ssh -o PreferredAuthentications=password -o PasswordAuthentication=yes -o PubkeyAuthentication=no project-pi
```

## Deployment

The Pi checkout is `/home/rrrodriguez13/projects/new-project`, on branch `main`.
This local repository already has a `pi` remote pointing to it. `setup-pi.sh`
records the repeatable provisioning steps and installs Git if needed.
For a different local repository, configure it once:

```powershell
git branch -M main
git remote add pi project-pi:/home/rrrodriguez13/projects/new-project
```

This repository records commits as `Adam <adam.fahs@berkeley.edu>`.
Its `main` branch tracks `pi/main`, so `git push` also deploys updates.
Git is configured locally to use its bundled OpenSSH client with the Windows SSH config;
Windows system OpenSSH transferred files but failed to finish the Git transport cleanly.
If the bundled Git installation moves, update the repository's `core.sshCommand` path.

Then edit and commit locally, and deploy:

```powershell
git add .
git commit -m "Describe the change"
git push pi main
```

Git updates the Pi checkout automatically using `receive.denyCurrentBranch=updateInstead`.
It rejects updates if tracked files on the Pi have been edited, and rejects rewritten history
and branch deletions. Develop locally; keep runtime data and secrets outside tracked files.
To roll back, use `git revert <commit>` locally, then push the new commit.

This deploys source files. Service startup, dependency installation, health checks, and
automatic restarts will be added when the application is defined. It does not install OS
updates. The Pi remote provides a second copy of Git history; an off-device hosted origin
can be added later for independent backup and collaboration.

Reserve `192.168.0.145` for the Pi in the router's DHCP settings so its address remains stable.

Reference: https://git-scm.com/docs/git-config#Documentation/git-config.txt-receivedenyCurrentBranch
