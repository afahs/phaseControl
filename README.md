# Raspberry Pi project foundation

Target: `rrrodriguez13@192.168.0.145`, SSH alias `project-pi`.
The application and runtime will be chosen later.

## SSH

The dedicated private key stays on this Windows computer at
`C:/Users/Adam/.ssh/id_ed25519_project_pi`; never commit or copy it to the Pi.
It has no passphrase for unattended updates and is protected by Windows file permissions.
The SSH alias uses public-key authentication only and the existing known host record.
After installing its public key on the Pi, test with `ssh project-pi`.
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
