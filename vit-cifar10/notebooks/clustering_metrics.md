# Silhouette, adjusted Rand, and normalized mutual information

These three scores answer two different questions about the same 10,000 class-token vectors in [`ClsAvgRndToken.ipynb`](ClsAvgRndToken.ipynb). The silhouette asks whether images of one class sit near each other and far from other classes. The adjusted Rand index and normalized mutual information ask whether a clustering that never sees the labels recovers those classes.

All three were computed on the L2-normalized 384-dimensional vectors. The UMAP plots are only a drawing of that space.

## Silhouette

For one image \(i\), let \(a(i)\) be its average distance to the other images of its own class. For every other class, compute the average distance from \(i\) to all images in that class. \(b(i)\) is the smallest of those averages. “Nearest” picks the class, and the distance to that class is still a mean over every image in it, not the distance to the single closest image. The score for that image is

\[
s(i)=\frac{b(i)-a(i)}{\max\{a(i),\,b(i)\}}.
\]

The reported silhouette is the average of \(s(i)\) over all 10,000 images. A per-class number is the average over the images of that class only.

\(s(i)\) is near 1 when the image is much closer to its own class than to any other class. It is near 0 when those two distances are about equal, so the image sits on a boundary. It is negative when the image is closer to some other class than to its own.

Take three airplanes at positions 0, 1, and 2 on a line, and two cars at 10 and 11. For the airplane at 0,

\[
a=\frac{|0-1|+|0-2|}{2}=1.5,\qquad b=\frac{|0-10|+|0-11|}{2}=10.5,
\]

\[
s=\frac{10.5-1.5}{10.5}=0.86.
\]

Move that airplane to position 9, between the two groups. Then \(a=(8+7)/2=7.5\) and \(b=(1+2)/2=1.5\), so \(s=(1.5-7.5)/7.5=-0.80\). It is in the airplane label, but it lives with the cars.

That is why the class token’s overall silhouette is only **0.15**, and why cat is **−0.003**. A typical test image is only slightly closer to its own class than to the next class, and a typical cat is as close to another class as to other cats. The classes are separable enough to see in a plot, and they are not tight blobs.

## Adjusted Rand index

The adjusted Rand index ignores distances. It compares two groupings of the same images: the true classes, and the 10 groups from k-means. The cluster numbers themselves do not matter. A group full of airplanes scores as a match even when k-means called that group cluster 7.

Look at one pair of images at a time. The labels and the clustering each make a yes-or-no decision: same group, or different groups. The pair is a hit when both decisions agree. Both say “together,” or both say “apart.”

Most pairs are an easy hit. With 10 classes, two random images are usually different classes, and two random images usually land in different clusters too. A useless clustering therefore still agrees on a large share of pairs. The raw hit rate, the Rand index, looks good for that reason.

The adjusted score removes that free agreement:

\[
\mathrm{ARI}=\frac{\text{hits} - \text{hits from a random clustering}}{\text{hits from a perfect clustering} - \text{hits from a random clustering}}.
\]

The random clustering is a shuffle that keeps the same group sizes. A perfect match scores 1. A shuffle scores 0. A grouping that splits real classmates on purpose scores below 0.

Four images, two airplanes and two cars:

- K-means keeps the two airplanes together and the two cars together. Every pair is a hit. ARI = 1.
- K-means pairs each airplane with a car. The two airplanes are split, and so are the two cars. That is worse than shuffling four images into two pairs. ARI = −0.5.

The class token’s ARI is **0.61**. The k-means groups recover a large part of the CIFAR-10 grouping, with real mistakes left. A value of 0 would mean those ten groups were no more related to the ten classes than a shuffle.

## Normalized mutual information

Normalized mutual information also compares the k-means partition with the true labels, and it also ignores distances. It asks how much the cluster id tells you about the class.

With the same contingency table, and \(p_{ij}=n_{ij}/n\),

\[
\mathrm{MI}=\sum_{ij}p_{ij}\log\frac{n\,n_{ij}}{a_i b_j},\qquad H(U)=-\sum_i\frac{a_i}{n}\log\frac{a_i}{n}.
\]

\(H(U)\) is the entropy of the class labels, and \(H(V)\) is the entropy of the cluster ids. The notebook uses scikit-learn’s default normalization, the arithmetic mean of those two entropies:

\[
\mathrm{NMI}=\frac{2\cdot\mathrm{MI}}{H(U)+H(V)}.
\]

The base of the logarithm cancels in the ratio. NMI is 1 when each cluster is pure and each class occupies one cluster. It is 0 when cluster id and class are independent.

On the same four images, the perfect grouping has NMI = 1. The crossed grouping \(\{1,3\}\) and \(\{2,4\}\) has NMI = 0: once you know the cluster, the image is equally likely to be either class. The adjusted Rand index was −0.5 for that same crossing, because pairs are systematically split and mutual information only reaches 0 when the two labelings share no information.

The class token’s NMI is **0.64**. Knowing the k-means cluster removes about two thirds of the uncertainty about the CIFAR-10 class.

## Why the three are reported together

The silhouette uses the geometry and the true labels. It can be low while the classes are still recoverable, which is this experiment: silhouette **0.15**, adjusted Rand **0.61**, NMI **0.64**. The classes overlap at their edges, so a typical point is only a little closer to its own class than to the next one, but the centers are far enough apart that 10-means still sorts most images into the right groups.

The adjusted Rand index and NMI use no geometry. They only check that the unsupervised partition matches the label partition, so a colorful UMAP is not being read as proof of cluster structure. The two scores can separate, as in the crossed example: mutual information says the labelings share no information, while the adjusted Rand index says the pairing is actively worse than chance. Here they agree, so the class token’s class structure is both geometrically mild and recoverable by clustering.

| Readout | Silhouette | Adjusted Rand | NMI |
|---|---:|---:|---:|
| Class token | 0.1464 | 0.6104 | 0.6434 |
| Mean of patch tokens | 0.0751 | 0.4673 | 0.5327 |
| First patch | 0.0085 | 0.2082 | 0.3067 |
