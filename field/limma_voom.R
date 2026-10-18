# voom-limma differential expression, CRC-like vs UC-like, with patient blocking.
#   Rscript limma_voom.R fit  counts.csv meta.csv formula out.csv
#   Rscript limma_voom.R boot counts.csv meta.csv formula out.csv module.csv n_boot seed
# 'boot' draws patients with replacement within group, reusing the observed filtering,
# TMM/voom weights and duplicate correlation, and records how often each gene is reselected
# by the same rule (BH < 0.05, |log2FC| > 1, top 150 by |log2FC|).
suppressMessages({library(limma); library(edgeR)})
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]
counts <- as.matrix(read.csv(args[2], row.names = 1, check.names = FALSE))
meta <- read.csv(args[3], row.names = 1, check.names = FALSE)[colnames(counts), , drop = FALSE]
form <- as.formula(args[4])
meta$group <- factor(meta$group, levels = c("UC-like", "CRC-like"))
for (v in intersect(c("cohort", "inflammation", "tissue_site"), all.vars(form))) meta[[v]] <- factor(meta[[v]])
design <- model.matrix(form, data = meta)

dge <- DGEList(counts)
dge <- dge[filterByExpr(dge, design), , keep.lib.sizes = FALSE]
dge <- calcNormFactors(dge)
v <- voom(dge, design)
rho <- duplicateCorrelation(v, design, block = meta$subject)$consensus

top_table <- function(E, w, dsg, block) {
  fit <- eBayes(lmFit(E, dsg, block = block, correlation = rho, weights = w))
  topTable(fit, coef = "groupCRC-like", number = Inf, sort.by = "none")
}

if (mode == "fit") {
  tt <- top_table(v, NULL, design, meta$subject)
  tt$gene <- rownames(tt)
  write.csv(tt[, c("gene", "logFC", "AveExpr", "t", "P.Value", "adj.P.Val")], args[5], row.names = FALSE)
  cat("genes tested:", nrow(tt), " duplicate correlation:", round(rho, 3), "\n")
} else {
  module <- read.csv(args[6])
  direction <- setNames(module$direction, module$gene)
  n_boot <- as.integer(args[7])
  E <- v$E; W <- v$weights; dimnames(W) <- dimnames(E); genes <- rownames(E)
  selected <- same_dir <- setNames(integer(length(genes)), genes)
  valid <- 0L
  groups <- aggregate(group ~ subject, meta, function(x) if (length(unique(x)) == 1) as.character(x[1]) else NA)
  groups <- groups[!is.na(groups$group), ]
  by_group <- split(groups$subject, groups$group)
  set.seed(as.integer(args[8]))
  for (b in seq_len(n_boot)) {
    draws <- unlist(lapply(names(by_group), function(g) sample(by_group[[g]], length(by_group[[g]]), replace = TRUE)))
    cols <- list(); m <- list()
    for (d in seq_along(draws)) {
      ids <- rownames(meta)[meta$subject == draws[d]]
      if (!length(ids)) next
      bm <- meta[ids, , drop = FALSE]; bm$subject <- paste0("draw", d)
      cols[[length(cols) + 1]] <- ids; m[[length(m) + 1]] <- bm
    }
    ids <- unlist(cols); bm <- do.call(rbind, m)
    bm$tissue_site <- droplevels(factor(bm$tissue_site))
    dsg <- try(model.matrix(form, data = bm), silent = TRUE)
    if (inherits(dsg, "try-error") || qr(dsg)$rank < ncol(dsg)) next
    tt <- try(top_table(E[, ids, drop = FALSE], W[, ids, drop = FALSE], dsg, bm$subject), silent = TRUE)
    if (inherits(tt, "try-error")) next
    valid <- valid + 1L
    eligible <- rownames(tt)[tt$adj.P.Val < 0.05 & abs(tt$logFC) > 1]
    top <- head(eligible[order(abs(tt[eligible, "logFC"]), decreasing = TRUE)], 150)
    selected[top] <- selected[top] + 1L
    shared <- intersect(rownames(tt), names(direction))
    agree <- shared[sign(tt[shared, "logFC"]) == direction[shared]]
    same_dir[agree] <- same_dir[agree] + 1L
  }
  write.csv(data.frame(gene = genes, selection_frequency = selected / valid,
                       direction_consistency = same_dir / valid, valid_bootstraps = valid),
            args[5], row.names = FALSE)
  cat("valid bootstrap replicates:", valid, "of", n_boot, "\n")
}
