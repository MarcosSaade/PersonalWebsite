import React from 'react';
import PageTemplate from '../components/PageTemplate';

import coverImage from '../images/neurocapture.png';
import patientsImage from '../images/neurocaptue/patients-module.png';

export default function NeuroCapturePage() {
  return (
    <PageTemplate title="NeuroCapture: A Desktop App for Multimodal Neurological Studies" image={coverImage}>
      <p>
        Dementia is one of the most pressing health challenges of our time, yet diagnosing it and monitoring its progression
        remains a complex and often subjective process. During my time at the Center for Microsystems and Biodesign, I built a platform to make
        real-world data collection simple, rigorous, and repeatable. The result is <strong>NeuroCapture</strong>—a full‑stack
        desktop application for creating rich, longitudinal, <em>multimodal</em> datasets that can power research into early detection
        and objective tracking of neurodegenerative disease.
      </p>

      <p>
        NeuroCapture was developed to streamline the entire research workflow:
        from enrolling a participant and recording speech or gait, to administering cognitive tests and exporting tidy data
        for analysis. It is designed for busy clinics: resilient to flaky Wi‑Fi, fast to operate, and opinionated about data
        integrity so researchers can focus on science rather than wrangling spreadsheets.
      </p>

      <img src={patientsImage} alt="NeuroCapture – Patients module with CSV export" className="page-image" />

      <h2>Why Multimodal Matters</h2>
      <p>
        Standard cognitive screens like MMSE or MoCA are helpful, but they often surface impairment only after significant
        decline. NeuroCapture augments those tools with <strong>speech</strong> and <strong>gait</strong> signals that allow earlier detection: subtle prosodic changes, acoustic markers, and motor signatures that are hard to notice in a short
        appointment but show up clearly in data.
      </p>

      <h2>What the App Includes</h2>
      <ul>
        <li><strong>Patient & Study Management</strong>: create participants with unique study identifiers; search, edit, and export
          cohorts. Strict referential integrity (FKs + cascade deletes) keeps related tables in sync.</li>
        <li><strong>Cognitive Assessments</strong>: research‑grade entry for MMSE/MoCA with per‑item subscores, validation, and free‑text
          clinical notes. Custom assessments are supported via a schema‑first config.</li>
        <li><strong>Speech Capture</strong>: record new audio in‑app or attach existing WAV/FLAC. Files are checksumed, normalized, and
          stored on disk with metadata in Postgres for fast queries.</li>
        <li><strong>One‑click Export</strong>: tidy CSVs joining demographics, assessments, and 150+ engineered speech features—ready for
          Python/R/Julia pipelines.</li>
      </ul>

      <h2>Under the Hood</h2>
      <h3>Architecture</h3>
      <p>
        The UI is an <strong>Electron + React</strong> app with Tailwind styling. It talks to a <strong>FastAPI</strong> service and a 
        <strong> PostgreSQL</strong> database (migrations via Alembic). The API is versioned and self‑documenting (Swagger/ReDoc). Data
        flows are designed to be resilient—writes are transactional; long‑running jobs (e.g., feature extraction) are queued
        and retried.
      </p>

      <h3>Audio Pipeline (research project)</h3>
      <p>
        For my speech‑analysis study, I built an automated pipeline: noise reduction, amplitude normalization, and voice
        activity detection, followed by feature extraction (prosodic timing/rhythm, spectral stats, MFCCs, jitter/shimmer,
        formants). The downstream ML handled class imbalance with <strong>SMOTE‑ENN</strong> and produced interpretable outputs via feature
        importance and post‑hoc explainers. This pipeline plugged directly into NeuroCapture’s export format to enable
        reproducible experiments on Spanish speech from DementiaBank‑style tasks.
      </p>

      <h3>Data Model</h3>
      <p>
        Core entities are <em>patients</em>, <em>assessments</em>, <em>audio recordings</em>, and <em>derived features</em>. Audio lives on the
        filesystem with SHA‑256 checks; the DB stores rich metadata, links, and stats. We use strict types, constraints, and
        audit timestamps to support longitudinal studies across multiple visits.
      </p>

      <h2>Design Goals</h2>
      <ul>
        <li><strong>Clinic‑friendly UX</strong>: fast keyboard workflows, clear empty states, and optimistic UI for common edits.</li>
        <li><strong>Reproducibility</strong>: deterministic preprocessing; exports that round‑trip into notebooks without manual cleaning.</li>
        <li><strong>Privacy</strong>: participant IDs instead of names; optional PHI redaction in exports; role‑based access for staff.</li>
        <li><strong>Extensibility</strong>: new modalities plug in via typed tables and background jobs without invasive UI rewrites.</li>
      </ul>

      <h2>Modules at a Glance</h2>
      <ul>
        <li><strong>Patients</strong>: searchable list, inline edit dialogs, and <em>Export CSV</em> (shown above).</li>
        <li><strong>Assessments</strong>: MMSE/MoCA itemization with automatic scoring and notes.</li>
        <li><strong>Audio</strong>: record/upload, waveform preview, processing status toasts, and feature summaries.</li>
        <li><strong>Admin</strong>: study configuration, user roles, and backup/restore.</li>
      </ul>

      <h2>What I Learned</h2>
      <p>
        Building NeuroCapture meant thinking end‑to‑end: UX for clinicians, failure modes in busy clinics, and data schemas
        that won’t collapse after a year of real usage. The project deepened my interest in multimodal inference and gave me
        a production‑quality base for future gait/vision work.
      </p>

      <p>        It was also a really good opportunity to learn some fullstack skills: I got to build a React app with Electron,
        design a Postgres schema, and implement a FastAPI backend. This was my first time working with backend code and SQL databases,
        which was a valuable learning experience.
      </p>

      <h2>Try It / Contact</h2>
      <p>
        If you’re a research group interested in multimodal data capture for neuro studies, feel free to reach out. I’m happy
        to demo the app, discuss the data model, or share the speech pipeline used in my dementia‑detection paper. You can also
        check out the app code on <a href="https://github.com/MarcosSaade/neurocapture">GitHub</a>, which is free to use and modify.
      </p>
    </PageTemplate>
  );
}
