import React from 'react';
import PageTemplate from '../components/PageTemplate';
import banorteachImage from '../images/finance.jpeg';
import uiPreview from '../images/banorteach/ui-preview.png';
import loginImg from '../images/banorteach/login.png';
import scenarioImg from '../images/banorteach/scenario.png';
import goalsImg from '../images/banorteach/goals.png';

export default function BanorTeachPage() {
  return (
    <PageTemplate title="BanorTeach: Personalized Financial Literacy Powered by AI" image={banorteachImage}>
      <p>
        Personal finance is one of the most practical skills a person can learn, yet it's often overlooked in formal education. Many people reach adulthood without ever being taught how to budget, save, or invest, leaving them vulnerable to debt and missed opportunities. At Hackathon Banorte 2024 (organized by a Mexican bank), my team and I set out to build a tool that could change that.
      </p>

      <p>
        The result was BanorTeach, a mobile app that uses AI to deliver personalized financial education. I developed the entire platform from scratch, including the frontend, backend, and AI integration, while my teammates supported the project with design input and pitch preparation. Our goal was to make financial literacy accessible, engaging, and tailored to each user's life context.
      </p>

      <h2>Why teach financial literacy</h2>
      <p>
        Financial literacy isn't just about knowledge—it's about agency. When people understand how money works, they're better equipped to avoid financial stress, plan for the future, and achieve personal goals. For banks, offering a tool like BanorTeach is also a strategic opportunity. It provides real value to customers, attracts new users, and encourages better financial behavior, which may reduce default rates or customer churn. Everyone benefits.
      </p>

      <p>
        In fact, BanorTeach could be offered as a free app by Banorte, with users redeeming rewards through their bank account. This ties education directly to real incentives while creating a positive feedback loop between customer engagement and financial wellness.
      </p>

      <img src={loginImg} alt="Login Page" className="page-image" />

      <h2>How the app works</h2>
      <p>
        When users first join, they are asked to provide a few basic details—such as their name, age, occupation, and financial goals. This information is used to adapt every part of the learning experience to their specific context.
      </p>

      <p>
        From there, users are taken to a dashboard where they can access different modules on key financial topics like saving, investing, and credit. Each module starts with clear, pre-written educational content, designed to build core understanding.
      </p>

      <p>
        If a user already feels confident in a topic, they can take a quiz to test out of the full module or skip directly to the interactive scenario at the end.
      </p>

      <img src={uiPreview} alt="Module Test-Out Quiz" className="page-image" />

      <p>
        The most unique feature comes at the end of each module: a personalized, AI-generated story (using gemini API). These are not just passive narratives, they are interactive, choose-your-own-adventure experiences. The story presents a realistic financial situation tailored to the user's profile, and at key moments, users are given multiple options for how to proceed. For example, a student learning about saving might face a decision between spending their monthly allowance on a concert or putting part of it aside for future expenses. A working adult might have to choose between taking a short-term loan or adjusting their budget. Each choice leads to a different outcome, and the storyline branches accordingly: smart financial decisions lead to better endings, while poor ones illustrate potential pitfalls.
      </p>

      <p>
        After the story concludes, the AI provides feedback on the user's choices, explaining which decisions were financially sound, which weren't, and why. This closing reflection helps reinforce the underlying concepts and gives users a safe environment to experiment, learn from mistakes, and develop better habits.
      </p>


      <img src={scenarioImg} alt="Personalized Financial Scenario" className="page-image" />

      <h2>Features at a glance</h2>

      <h3>Personalized lessons</h3>
      <p>
        Every piece of content—explanations, examples, and stories—is adapted to the user’s profile. The goal is to make financial education feel relevant and actionable, not abstract or one-size-fits-all.
      </p>

      <h3>Goal tracking</h3>
      <p>
        Users can set concrete financial goals, like saving for a vacation or paying off a loan. The app helps them monitor progress and stay motivated.
      </p>

      <img src={goalsImg} alt="Goal Tracking Feature" className="page-image" />

      <h3>Gamified experience</h3>
      <p>
        Learning is reinforced through points, progress tracking, and achievement systems. Completing modules or making good decisions in simulations earns users rewards and encourages continued engagement.
      </p>

      <h2>Behind the scenes</h2>

      <h3>Frontend</h3>
      <p>
        The app was built using React Native, allowing for a cross-platform experience with shared components and a consistent, clean design. I focused on creating an interface that felt simple and welcoming to all users.
      </p>

      <h3>Backend</h3>
      <p>
        The backend runs on Node.js with an Express server. It handles user registration, stores learning progress, and routes requests to the AI engine. The architecture is modular, allowing for easy future expansion.
      </p>

      <h3>AI integration</h3>
      <p>
        I used Google Cloud’s Vertex AI to generate the personalized learning stories. This required careful prompt design to ensure the output aligned with pedagogical goals, stayed grounded in realistic financial behavior, and adjusted dynamically based on user input. The result was content that felt alive and personal.
      </p>

      <h2>Challenges and insights</h2>
      <ul>
        <li>Time constraints forced tight prioritization and fast iteration</li>
        <li>Managing the full tech stack—frontend, backend, AI—was intense but satisfying</li>
        <li>UI design had to communicate complex ideas without overwhelming users</li>
      </ul>

      <h2>Looking forward</h2>
      <ul>
        <li>Offer real rewards redeemable through Banorte accounts to boost user engagement</li>
        <li>Add more advanced modules as users progress—from credit scores to taxes to investing strategy</li>
        <li>Implement behavior-based recommendations that adapt over time</li>
        <li>Add social and community features such as group challenges or shared goals</li>
        <li>Use analytics to help users visualize their learning journey and spot strengths or gaps</li>
      </ul>

      <h2>Reflection</h2>
      <p>
        BanorTeach was built in a single weekend, but the idea behind it has lasting relevance. Personalized education powered by AI has the potential to radically improve how people learn skills that impact their lives every day. By grounding lessons in real-world situations and aligning them with each user’s goals, we make financial education more human, not just more efficient.
      </p>

      <p>
        I’m proud of what I accomplished with this project. It was a chance to design and build something meaningful, and it deepened my conviction that good software can create real-world change.
      </p>

      <p>
        You can explore the code and see the project in more detail on <a href="https://github.com/MarcosSaade/BanorTeach" target="_blank" rel="noopener noreferrer">GitHub</a>.
      </p>
    </PageTemplate>
  );
}
