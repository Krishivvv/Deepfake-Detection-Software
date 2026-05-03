"use client";

import { useState } from "react";

const FAQ = [
  {
    q: "What dataset is the model trained on?",
    a: "A 1 000-video subset of FaceForensics++, split 700 / 150 / 150 across train, val, and test. The fake class spans deepfakes, face2face, faceswap, and neuraltextures manipulations. Real videos come from the FaceForensics++ originals split.",
  },
  {
    q: "Will it catch deepfakes from a manipulation it wasn't trained on?",
    a: "Generalisation to unseen manipulations is a known weakness of any model trained on a single forgery family. Veridex was trained on four — if a new manipulation shares low-level statistics with one of them, the model often catches it. If it's a fundamentally different generation pipeline, expect lower confidence and higher error rates.",
  },
  {
    q: "Why is real-class recall lower than fake-class recall?",
    a: "The dataset is 4 : 1 fake : real, with only 140 unique real training videos. WeightedRandomSampler and threshold calibration close most of the gap (real-recall is 73 % at the deployed threshold, up from 5.5 % at the default 0.5), but more real data is the only durable fix.",
  },
  {
    q: "Where do uploads go?",
    a: "They stay on your machine. The Flask backend writes the file to a temp directory under app/static/uploads/, runs the model, and deletes the file on the way out. Nothing is sent to a third-party API.",
  },
  {
    q: "How fast is one prediction?",
    a: "About 15 seconds end-to-end on a 2-core 7th-gen i5 CPU — dominated by ResNet-50 forward over 32 face crops. On a GPU it would be sub-second; the model is loaded once at server startup, so there is no per-request load cost.",
  },
  {
    q: "Can I retrain the model on a new dataset?",
    a: "Yes. The repo's TRAINING_GUIDE walks through both the CPU path (cache features once, train just the head) and the GPU path (full end-to-end fine-tune on Colab) with exact commands.",
  },
  {
    q: "Is this fit for forensic use?",
    a: "No — and we say so on the About page. Veridex is a course / research demo. Predictions are probabilistic, real-recall is not 100 %, and the model has only been evaluated on FaceForensics++.",
  },
];

export function FAQAccordion() {
  const [open, setOpen] = useState<number | null>(0);

  return (
    <section id="faq" className="container-page py-20 sm:py-28 scroll-mt-page">
      <div className="grid gap-12 lg:grid-cols-[1fr_2fr] lg:gap-20 items-start">
        <div>
          <span className="chip">FAQ</span>
          <h2 className="heading-lg mt-4 text-mint-50">
            Questions you'll probably ask.
          </h2>
          <p className="mt-4 text-ink-300 leading-relaxed">
            If something is missing here, the README and HANDOFF in the repo
            have a deeper trace of the design decisions.
          </p>
        </div>

        <div className="space-y-3">
          {FAQ.map((item, idx) => {
            const isOpen = open === idx;
            return (
              <div
                key={item.q}
                className={[
                  "rounded-2xl border transition-all duration-200",
                  isOpen
                    ? "border-mint-500/40 bg-ink-700"
                    : "border-ink-500/70 bg-ink-700/60 hover:border-mint-500/30",
                ].join(" ")}
              >
                <button
                  type="button"
                  className="flex w-full items-center justify-between gap-4 text-left px-5 py-4"
                  onClick={() => setOpen(isOpen ? null : idx)}
                  aria-expanded={isOpen}
                >
                  <span className="font-medium text-mint-50">{item.q}</span>
                  <span
                    className={[
                      "shrink-0 h-7 w-7 inline-flex items-center justify-center rounded-full border transition-transform",
                      isOpen
                        ? "border-mint-500/50 text-mint-300 rotate-45"
                        : "border-ink-500 text-ink-300",
                    ].join(" ")}
                    aria-hidden
                  >
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="h-4 w-4"
                    >
                      <path d="M12 5v14" />
                      <path d="M5 12h14" />
                    </svg>
                  </span>
                </button>
                <div
                  className={[
                    "overflow-hidden transition-all duration-300",
                    isOpen ? "max-h-72" : "max-h-0",
                  ].join(" ")}
                >
                  <p className="px-5 pb-5 text-sm text-ink-300 leading-relaxed">
                    {item.a}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
