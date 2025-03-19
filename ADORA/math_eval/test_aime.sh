CUDA_VISIBLE_DEVICES=0 python test_aime.py --stage 1 --step 120 > log/aime/1/120.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python test_aime.py --stage 1 --step 180 > log/aime/1/180.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python test_aime.py --stage 1 --step 300 > log/aime/1/300.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python test_aime.py --stage 1 --step 360 > log/aime/1/360.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python test_aime.py --stage 1 --step 420 > log/aime/1/420.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python test_aime.py --stage 1 --step 480 > log/aime/1/480.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python test_aime.py --stage 1 --step 600 > log/aime/1/600.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python test_aime.py --stage 1 --step 660 > log/aime/1/660.log 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python test_aime.py --stage 1 --step 720 > log/aime/1/720.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python test_aime.py --stage 1 --step 780 > log/aime/1/780.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python test_aime.py --stage 1 --step 900 > log/aime/1/900.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python test_aime.py --stage 1 --step 960 > log/aime/1/960.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python test_aime.py --stage 1 --step 1020 > log/aime/1/1020.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python test_aime.py --stage 1 --step 1080 > log/aime/1/1080.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python test_aime.py --stage 1 --step 1320 > log/aime/1/1320.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python test_aime.py --stage 1 --step 1380 > log/aime/1/1380.log 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python test_aime.py --stage 1 --step 1440 > log/aime/1/1440.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python test_aime.py --stage 1 --step 1560 > log/aime/1/1560.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python test_aime.py --stage 1 --step 1620 > log/aime/1/1620.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python test_aime.py --stage 1 --step 1680 > log/aime/1/1680.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python test_aime.py --stage 1 --step 1740 > log/aime/1/1740.log 2>&1 &