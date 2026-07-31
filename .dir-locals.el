((org-mode
  (after-save-hook . (lambda () (org-latex-export-to-latex)))))
