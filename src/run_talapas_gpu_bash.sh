#!/usr/bin/env bash

# Get a bash console with one GPU
srun --pty --gres=gpu:1 --mem=24G --time=1200 --partition=gpu bash
