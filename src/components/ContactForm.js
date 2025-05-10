import React from 'react';
import './ContactForm.css';

function ContactForm() {
  return (
    <section id="contact" className="contact-form">
      <h2>Contact Me</h2>
      <form action="https://formspree.io/f/{your_form_id}" method="POST">
        <input type="text" name="name" placeholder="Your Name" required />
        <input type="email" name="_replyto" placeholder="Your Email" required />
        <textarea name="message" placeholder="Your Message" rows="6" required></textarea>
        <button type="submit">Send Message</button>
      </form>
    </section>
  );
}

export default ContactForm;
