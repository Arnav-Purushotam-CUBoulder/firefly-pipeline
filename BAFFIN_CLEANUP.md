# Baffin account cleanup

Log in:

```bash
ssh IDENTIKEY@baffin.colorado.edu
```

Disable shell history for this session before cleanup:

```bash
export HISTFILE=/dev/null
set +o history
history -c
```

Cancel any jobs still running:

```bash
squeue --me
scancel -u "$USER"
```

Inspect what is currently in your account:

```bash
du -sh "$HOME" /scratch/isilon/"$USER" 2>/dev/null
find "$HOME" -mindepth 1 -maxdepth 1 -printf '%P\n' | sort
find /scratch/isilon/"$USER" -mindepth 1 -maxdepth 1 -printf '%P\n' | sort
```

Delete everything in your personal scratch directory:

```bash
cd /
find /scratch/isilon/"$USER" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
```

Delete everything in your home directory except `.snapshot`:

```bash
find "$HOME" -mindepth 1 -maxdepth 1 ! -name '.snapshot' -exec rm -rf -- {} +
```

Verify both locations are empty:

```bash
find "$HOME" -mindepth 1 -maxdepth 1 ! -name '.snapshot' -print
find /scratch/isilon/"$USER" -mindepth 1 -maxdepth 1 -print
du -sh "$HOME" /scratch/isilon/"$USER" 2>/dev/null
```

Exit:

```bash
exit
```

Notes:

- This removes hidden files too, including `~/.ssh` and virtual environments under your home directory.
- Baffin docs say home and project storage use snapshots, so deleted data may still count temporarily until snapshot expiration.
- If you need confirmation that the account is fully cleared, contact `bit-help@colorado.edu`.
