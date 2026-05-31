#bash
git worktree list --porcelain | grep '^worktree ' | grep -v $(pwd) | cut -d' ' -f2 | xargs -I {} git worktree remove '{}' && git branch --list 'kb/*' | sed 's/^..//' | xargs git branch -D
