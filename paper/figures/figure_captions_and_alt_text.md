# Paper figure captions and accessibility descriptions

## Figure 1

**Caption.** Figure 1. Study design for auditing a JARVIS-DFT pretrained ALIGNN formation-energy checkpoint across an oxide comparator and nitride target. The same fixed family test sets are used for zero-shot evaluation and partial fine-tuning; matched from-scratch baselines are available only at N = 50 and N = 500. Family definitions are intentionally asymmetric: the oxide arm is O-bearing and includes O+N structures, whereas the nitride arm is N-bearing and O-free.

**Alt text.** Workflow diagram showing one JARVIS-DFT pretrained ALIGNN checkpoint evaluated on O-bearing oxide and O-free nitride families through zero-shot, partial fine-tuning at six training sizes, matched from-scratch baselines at N = 50 and 500, and frozen-embedding analysis.

**Long description.** The shared pretrained checkpoint branches to oxide and nitride fixed-test arms, whose exact checkpoint-training JID overlap is zero. Both arms feed zero-shot evaluation and partial fine-tuning of the last GCN block plus output head at N = 10, 50, 100, 200, 500, and 1,000. From-scratch comparison is restricted to N = 50 and 500. The workflow produces audited error, data-efficiency, initialization, and representation evidence.

## Figure 2

**Caption.** Figure 2. Zero-shot chemical-family performance and Protocol 1 fine-tuning curves. (a) Bars show fixed-test mean absolute error (MAE); error bars are 95% percentile intervals from 50,000 structure-level bootstrap replicates. (b) Open markers show individual fine-tuning seeds and lines show arithmetic five-seed means under Protocol 1; horizontal lines show family-specific zero-shot MAE. Every Protocol 1 mean remains above its corresponding zero-shot line, and the nitride mean reaches its maximum at N = 200.

**Alt text.** Two-panel plot: nitride zero-shot MAE is about twice oxide with nonoverlapping bootstrap intervals, while corrected five-seed Protocol 1 curves remain above both family zero-shot references and nitride peaks at N = 200.

**Long description.** Panel a compares oxide MAE 0.0342 and nitride MAE 0.0695 eV per atom with structure-bootstrap intervals. Panel b shows all five seed results at six training sizes for each family, their mean curves, and the two zero-shot references. The oxide curve declines toward but does not reach its zero-shot level after N = 50. The nitride curve is non-monotonic and peaks at N = 200 before declining at larger N.

## Figure 3

**Caption.** Figure 3. Pretrained-initialization benefit and selected-checkpoint depth under Protocol 1. (a) Seed points and five-seed mean ± sample standard deviation compare fine-tuned and from-scratch MAE at N = 50 and N = 500 on a logarithmic scale. (b) Seed points and mean lines show the validation-selected checkpoint epoch. An epoch-1 selection denotes the end-of-epoch-1 checkpoint and does not establish byte identity with the zero-shot checkpoint or absence of parameter updates.

**Alt text.** Two-panel plot: fine-tuned models have far lower MAE than random initialization at N = 50 and 500, while oxide selects later checkpoints from N = 50 and nitride is epoch-1 through N = 100, mixed at N = 200, and later at N at least 500.

**Long description.** Panel a shows a large gap between fine-tuned and from-scratch test MAE for both families and both supported sizes. Panel b shows five selected epochs per family and size plus their means. Oxide seeds select checkpoints later than epoch 1 from N = 50 onward. Nitride seeds all select epoch 1 through N = 100, are heterogeneous at N = 200 with epochs 49, 1, 1, 1, and 1, and all select later checkpoints at N = 500 and 1,000.

## Figure 4

**Caption.** Figure 4. Frozen-representation geometry and canonical within-nitride distance–error association. (a) Descriptive standardized principal-component projection of fixed-test `last_alignn_pool` embeddings, with the basis fitted on the balanced train–validation pool. (b) Canonical nitride absolute zero-shot error versus mean Euclidean distance to the five nearest oxide-reference embeddings in the raw 256-dimensional space. The positive Spearman association is protocol-specific and correlational; it does not identify a causal mechanism.

**Alt text.** Two-panel plot: a descriptive PCA separates many oxide and nitride frozen embeddings, and canonical nitride error tends to increase with 5-nearest-oxide embedding distance, with Spearman rho about 0.346 and a positive 95% interval.

**Long description.** Panel a projects 1,484 oxide and 242 nitride fixed-test embeddings into a common PCA basis fitted without test labels on the balanced pool. Panel b plots 242 nitride structures using canonical zero-shot errors and raw-space 5-nearest-neighbor distances to the oxide reference pool. The recomputed Spearman correlation is 0.346 with 95% bootstrap interval 0.225 to 0.463 and BH-FDR q approximately 0.000150. These results are an association under the recorded representation and distance protocol, not a causal explanation.
