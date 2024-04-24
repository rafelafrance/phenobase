#!/bin/zsh

SESSION="phenobase"
PHENO="$HOME/work/phenology"

tmux new -s $SESSION -d
tmux rename-window -t $SESSION pheno
tmux send-keys -t $SESSION "cd $PHENO/phenobase" C-m
tmux send-keys -t $SESSION "vrun .venv" C-m
tmux send-keys -t $SESSION "git status" C-m

tmux new-window -t $SESSION
tmux rename-window -t $SESSION pheno_zsh
tmux send-keys -t $SESSION "cd $PHENO/phenobase" C-m
tmux send-keys -t $SESSION "vrun .venv" C-m

tmux new-window -t $SESSION
tmux rename-window -t $SESSION herbarium
tmux send-keys -t $SESSION "cd $PHENO/herbarium" C-m
tmux send-keys -t $SESSION "vrun .venv" C-m
tmux send-keys -t $SESSION "git status" C-m

tmux new-window -t $SESSION

tmux select-window -t $SESSION:1
tmux attach -t $SESSION
