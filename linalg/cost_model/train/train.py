import random
from math import ceil
import numpy as np
import torch
from torch import nn
from collections import Counter


class Dataset:
    def __init__(self, data, kernels, platform="cortex_a53"):
        self.data = data
        self.kernels = kernels
        self.platform = platform

    @classmethod
    def from_tract_outputs(cls, input_file, platform="cortex_a53"):
        kernels = set()
        mkn = set()
        params = []
        for line in open(input_file):
            x = line.strip().split()
            if len(x) < 7:
                continue
            m, k, n = list(map(float, x[3:6]))
            dur = float(x[-1])
            kernels.add((x[0], int(x[1]), int(x[2])))
            mkn.add((m, k, n))
            params.append((x[0], m, k, n, dur))
        kernels = sorted(list(kernels))
        kernel_names = list(map(lambda ker: ker[0], kernels))
        mkn_kernels = {mkn_value: [0 for k in kernels] for mkn_value in mkn}
        for krn, m, k, n, dur in params:
            i = kernel_names.index(krn)
            mkn_kernels[(m, k, n)][i] = dur
        sorted_mkn = sorted(list([(m * k * n, (m, k, n)) for m, k, n in mkn]))

        data = []
        for pdt, (m, k, n) in sorted_mkn:
            if pdt == 0:
                continue
            data.append((np.array([m, k, n]), np.array(mkn_kernels[(m, k, n)])))
        return cls(data, kernels)

    def shuffle(self):
        random.shuffle(self.data)

    def split(self, validation=0.1, shuffle=True):
        if shuffle:
            self.shuffle()
        N = len(self.data)
        S = ceil(validation * N)
        trainset = Dataset(self.data[S:], self.kernels)
        valset = Dataset(self.data[:S], self.kernels)
        return trainset, valset

    def filter_mkn(self, max_val=None, min_val=None):
        def keep(x):
            mkn = np.prod(x)
            if max_val is not None and mkn > max_val:
                return False
            if min_val is not None and mkn < min_val:
                return False
            return True

        data = [(x, y) for x, y in self.data if keep(x)]
        return Dataset(data, self.kernels)

    def __len__(self):
        return len(self.data)

    def get_mr_nr_values(self):
        _, mrs, nrs = zip(*self.kernels)
        return sorted(list(set(mrs))), sorted(list(set(nrs)))

    def get_classif_features_for(self, x):
        m, k, n = x

        mrs, nrs = self.get_mr_nr_values()
        fts = [
            np.log(m),
            np.log(k),
            np.log(n),
            np.log(m * k * n),
        ]
        for mr in mrs:
            fts.append(m % mr)
            fts.append(float(m % mr != 0))
        for nr in nrs:
            fts.append(n % nr)
            fts.append(float(n % nr != 0))

        return np.array(fts)

    def get_classif_features(self, soft_targets=True, temp=5e-2):
        feats = []
        targets = []
        for x, y in self.data:
            x_features = self.get_classif_features_for(x)
            if soft_targets:
                tgt = 1 / y
                tgt = tgt - tgt.max()
                tgt *= temp
                tgt = np.exp(tgt)
                tgt = tgt / tgt.sum()
                targets.append(tgt)
            else:
                targets.append(y.argmin())
            feats.append(x_features)
        return np.array(feats), np.array(targets)

    def get_rel_diffs(self, preds):
        diffs = []
        for (x, y), z in zip(self.data, preds):
            t = y.argmin()
            if z == t:
                continue
            diffs.append((y[z] - y.min()) / y.min())
        return np.array(diffs)

    def big_product_behaviour(self):
        sorted_data = sorted((np.prod(x), np.argmin(y)) for x, y in self.data)
        biggest_exp = sorted_data[:-int(len(self) / 100)]
        _, choices = zip(*biggest_exp)
        kernel_ix = Counter(choices).most_common(1)[0][0]
        kernel = self.kernels[kernel_ix][0]
        threshold = sorted_data[-1][0]
        return (threshold, kernel)

class MLP(nn.Module):
    def __init__(
        self,
        kernels,
        num_features,
        num_hiddens,
        normalize=True,
        num_updates=3000,
        batch_size=128,
        weight_decay=0.0001,
        soft_preds=False,
    ):
        super().__init__()
        self.kernels = kernels
        num_kernels = len(kernels)
        self.linear_1 = nn.Linear(num_features, num_hiddens)
        self.act = nn.Tanh()
        self.linear_2 = nn.Linear(num_hiddens, num_kernels)
        self.softmax = nn.LogSoftmax(dim=1)
        self.mean = None
        self.std = None
        self._normalize = normalize
        self.num_updates = num_updates
        self.batch_size = batch_size
        self.soft_preds = soft_preds
        self.weight_decay = weight_decay

    def forward(self, x):
        y1 = self.linear_1.forward(x)
        y = self.act.forward(y1)

        y = self.linear_2.forward(y)
        return self.softmax.forward(y)

    def normalize(self, X):
        if self._normalize:
            return (X - self.mean) / self.std
        return X

    def predict_proba(self, x):
        x = self.normalize(x)
        tx = torch.from_numpy(x).float()
        y = self.forward(tx)
        return np.exp(y.detach().numpy())

    def predict(self, x):
        y = self.predict_proba(x)
        return y.argmax(axis=1)

    def fit(self, X, y):
        if self._normalize:
            self.mean = X.mean(axis=0, keepdims=True)
            self.std = X.std(axis=0, keepdims=True)
            self.std[self.std < 1e-4] = 1e-4
            X = self.normalize(X)

        updates = 0
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-3, weight_decay=self.weight_decay
        )
        loss = (
            torch.nn.KLDivLoss(reduction="batchmean")
            if self.soft_preds
            else torch.nn.NLLLoss()
        )
        indices = list(range(X.shape[0]))
        num_batches = len(indices) // self.batch_size

        prev_loss = None
        num_iter_no_impr = 0

        while updates < self.num_updates:
            random.shuffle(indices)
            total_loss = 0
            batches_seen = 0
            for bnum in range(num_batches):
                bb = self.batch_size * bnum
                be = bb + self.batch_size
                Xb = X[indices[bb:be]]
                yb = y[indices[bb:be]]

                tx = torch.from_numpy(Xb).float()
                if self.soft_preds:
                    ty = torch.from_numpy(yb).float()
                else:
                    ty = torch.from_numpy(yb).long()

                optimizer.zero_grad()
                z = self.forward(tx)
                loss_val = loss(z, ty)
                loss_val.backward()
                optimizer.step()

                sloss = loss_val.detach().numpy()
                total_loss += sloss

                updates += 1
                batches_seen += 1
                if updates > self.num_updates:
                    break

            total_loss /= batches_seen
            if prev_loss is not None:
                impr = (prev_loss - total_loss) / prev_loss
                if impr < 1e-4:
                    num_iter_no_impr += 1
                else:
                    num_iter_no_impr = 0
            prev_loss = total_loss
            if num_iter_no_impr > 4:
                break

def save_as_rust(mlp, dataset, output):
    with open(output, 'w') as f:
        mrs, nrs = dataset.get_mr_nr_values()
        params = {}
        for name, tensor in mlp.named_parameters():
            params[name] = tensor.detach().numpy()
        big_product_mkn_threshold, big_product_kernel_choice = dataset.big_product_behaviour()
        f.write(f"""use crate::frame::mmm::CostModel;
        pub fn model() -> CostModel<'static> {{
            CostModel {{
                big_product_mkn_threshold: {big_product_mkn_threshold},
                big_product_kernel_choice: "{big_product_kernel_choice}",
                kernels: &{str(list(map(lambda k: k[0], mlp.kernels))).replace("'", '"')},
                mrs: &{mrs},
                nrs: &{nrs},
                feat_norm_mean: &{mlp.mean.flatten().tolist()},
                feat_norm_stddev: &{mlp.std.flatten().tolist()},
                w1: &{params["linear_1.weight"].flatten().tolist()},
                b1: &{params["linear_1.bias"].flatten().tolist()},
                w2: &{params["linear_2.weight"].flatten().tolist()},
                b2: &{params["linear_2.bias"].flatten().tolist()},
            }}
        }}
""")

def train_one_mlp(
    dataset, hidden_layer_size, validation=0.2, num_updates=3000,
):
    train_ds, dev_ds = dataset.split(validation=validation)
    Xtrain, ytrain = train_ds.get_classif_features()
    clf = MLP(
        train_ds.kernels,
        Xtrain.shape[1],
        hidden_layer_size,
        soft_preds=True,
        num_updates=num_updates,
    )
    clf.fit(Xtrain, ytrain)

    Xtest, ytest = dev_ds.get_classif_features()
    ztest = clf.predict(Xtest)
    ptest = clf.predict_proba(Xtest)

    gtclass = ytest.argmax(axis=1)
    accuracy = 100 * np.sum(gtclass == ztest) / len(ytest)
    crank = [
        len(dev_ds.kernels) - list(stdpreds).index(gty)
        for stdpreds, gty in zip(ptest.argsort(axis=1), gtclass)
    ]
    rdiffs = list(dev_ds.get_rel_diffs(ztest))
    return accuracy, rdiffs, crank, clf


PASS_TESTS = {
    "cortex_a7": [
        ([16, 60, 8], "armv7neon_mmm_f32_8x4_cortexa7"),
        ([16, 64, 8], "armv7neon_mmm_f32_8x4_cortexa7"),
#         ([2, 32, 8], "generic_f32_4x4"),
        ([64, 48, 8], "armv7neon_mmm_f32_8x4_cortexa7"),
        ([256, 768, 6], "armv7neon_mmm_f32_8x6_cortexa7"),
        ([512, 1536, 18], "armv7neon_mmm_f32_8x6_cortexa7"),
        ([512, 1536, 24], "armv7neon_mmm_f32_8x6_cortexa7"),
    ],
    "cortex_a9": [
        ([16, 60, 8], "armv7neon_mmm_f32_8x4_cortexa9"),
        ([16, 64, 8], "armv7neon_mmm_f32_8x4_cortexa9"),
#         ([2, 32, 8], "generic_f32_4x4"),
        ([64, 48, 8], "armv7neon_mmm_f32_8x4_cortexa9"),
        ([256, 768, 6], "armv7neon_mmm_f32_8x6_cortexa9"),
    ],
    "cortex_a53": [
        ([16, 60, 8], "arm64simd_mmm_f32_8x8_a53"),
        ([16, 64, 8], "arm64simd_mmm_f32_8x8_a53"),
        ([2, 32, 8], "arm64simd_mmm_f32_8x8_a53"),
        ([64, 48, 8], "arm64simd_mmm_f32_8x8_a53"),
        ([256, 768, 4], "arm64simd_mmm_f32_24x4_a53"),
    ],
    "cortex_a55": [
        ([16, 60, 8], "arm64simd_mmm_f32_8x8_a53"),
        ([16, 64, 8], "arm64simd_mmm_f32_8x8_a53"),
        ([64, 48, 8], "arm64simd_mmm_f32_8x8_a53"),
        ([256, 768, 4], "arm64simd_mmm_f32_16x4_a55"),
        ([512, 1536, 24], "arm64simd_mmm_f32_12x8_a55"),
        ([60, 40, 1337], "arm64simd_mmm_f32_12x8_a55"),
        ([72, 2400,728], "arm64simd_mmm_f32_12x8_a55"),
    ],
}


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Train a neural network to predict which mmm kernel"
    )
    parser.add_argument(
        "-H",
        "--hidden-size",
        type=int,
        default=40,
        help="Size of the hidden layer of the neural network [default: 40].",
    )
    parser.add_argument(
        "-N", "--num-trainings", type=int, default=1, help="Number of trainings.",
    )
    parser.add_argument("--platform", default="cortex_a53", choices=PASS_TESTS.keys(), help="Platform")
    parser.add_argument("dataset")
    parser.add_argument("output_rs")
    args = parser.parse_args()
    print("Loading dataset...")
    dataset = Dataset.from_tract_outputs(args.dataset, platform=args.platform)
    print(f"Loaded {len(dataset)} samples")
    passed = False
    tests = PASS_TESTS[args.platform]
    best = None
    best_acc = 0.0
    trained = 0
    while not passed:
        print(f"[{trained + 1}] Training MLP with {args.hidden_size} units...")
        accuracy, _, _, model = train_one_mlp(dataset, args.hidden_size)
        trained += 1
        num_passed = 0
        for mkn, ker in tests:
            x = dataset.get_classif_features_for(mkn)
            y = model.predict([x])[0]
            if dataset.kernels[int(y)][0] == ker:
                num_passed += 1
            else:
                print(f"for {mkn} predicted: {dataset.kernels[int(y)]}, expected: {ker}")
        passed = num_passed == len(tests)
        if passed and accuracy > best_acc:
            best_acc = accuracy
            best = model
        color = 92 if passed else 91
        print(
            f"\tAccuracy: {accuracy:.1f}% ... \033[{color}mPASSED {num_passed} / {len(tests)}\033[0m"
        )
        passed = passed and trained >= args.num_trainings
    print(f"Saving model to {args.output_rs}")
    save_as_rust(best, dataset, args.output_rs)


if __name__ == "__main__":
    main()
