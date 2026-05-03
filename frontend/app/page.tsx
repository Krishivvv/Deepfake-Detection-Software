import { Hero } from "@/components/Hero";
import { StatBar } from "@/components/StatBar";
import { UseCaseCards } from "@/components/UseCaseCards";
import { HowItWorks } from "@/components/HowItWorks";
import { FeatureGrid } from "@/components/FeatureGrid";
import { FAQAccordion } from "@/components/FAQAccordion";

export default function HomePage() {
  return (
    <>
      <Hero />
      <StatBar />
      <UseCaseCards />
      <HowItWorks />
      <FeatureGrid />
      <FAQAccordion />
    </>
  );
}
