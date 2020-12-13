
function make_shapeworld {
    python shapeworld.py
}

function lm {
    python language_model.py --dataset shapeworld --cuda --debug | tee lm.log
}

function l0 {
    python train.py --dataset shapeworld --l0 --cuda --debug --lr 0.001 | tee l0.log
}

function s0 {
    python train.py --dataset shapeworld --s0 --cuda --debug --lr 0.001 | tee s0.log
}

function al {
    python train.py --dataset shapeworld --amortized --activation gumbel \
        --lr 0.001 \
        --cuda --debug \
        --penalty length --batch_size 128 \
        | tee amortized-length.log
}

function ab {
    python train.py --dataset shapeworld --amortized --activation gumbel \
        --lr 0.001 \
        --batch_size 128 \
        --cuda --debug \
        --penalty bayes \
        | tee amortized-bayes.log
}

function ev {
    python train.py --dataset shapeworld --cuda --eval_only
}

