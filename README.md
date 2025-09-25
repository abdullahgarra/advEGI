# advEGI
Adversarial edge edits + EGI experiments.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9b78334f-000a-4e9c-aaef-e1adfe071fe7" width="45%" />
  <img src="https://github.com/user-attachments/assets/43e35201-c0b6-4796-8cdd-21f50afba27c" width="45%" />
</p>



 This is a small research codebase exploring whether tiny edge edits (adversarial, PGD-style “score-and-flip”) help Ego-Graph Infomax (EGI) transfer across graphs on the classic airport-role benchmarks.

**What we looked at:**
- Train EGI as usual vs. EGI with light adversarial edge edits.
- Check if simple degree/centrality explains most of the performance.
- Try different source graphs (original Europe, degree-preserving shuffle, random) and see what actually transfers best.

**What we found:**
- Adversarial edge edits give small/mixed gains: sometimes a nudge (e.g., EU→USA), sometimes not. (and the procedure is very noisy)
- Degree/centrality carries a big chunk of the signal on these benchmarks (especially Brazil).
- The usual “pick the closest source by structure (ΔD)” **isn’t always a safe bet** here.

**Why the pictures?**  
Left: how EGI thinks—compare distributions of local ego-graphs across domains.  
Right: classic image adversarial example—tiny changes can flip a prediction; adversarial training tries to make models ignore such noise.

