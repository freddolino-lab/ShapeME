use lettre::{
    transport::smtp::authentication::Credentials,
    AsyncSmtpTransport,
    AsyncTransport,
    Message,
    Tokio1Executor,
};
use std::error::Error;

pub async fn mail(job_id: &str, name: &str, address: &str) -> Result<(), Box<dyn Error>> {
    let smtp_credentials = Credentials::new(
        "smtp_username".to_string(),
        "smtp_password".to_string(),
    );

    let mailer = AsyncSmtpTransport::<Tokio1Executor>::relay("smtp.email.com")?
        .credentials(smtp_credentials)
        .build();

    let from = "Hello World <hello@world.com>";
    let to = format!("{} <{}>", name, address);
    let subject = format!("ShapeME Job {}", job_id);
    let body = format!(
        "Hello. <br/>
        Your ShapeME job (id: {id}) has been successfully submitted.
        Upon job completion you will receive an email.
        In the meantime, to check the status
        of your job, visit <href=\"localhost:8000/jobs/{id}\">this site</href>."
        id = job_id,
    );

    send_email_smtp(&mailer, from, to, subject, body).await
}

async fn send_email_smtp(
        mailer: &AsyncSmtpTransport<Tokio1Executor>,
        from: &str,
        to: &str,
        subject: &str,
        body: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let email = Message::builder()
        .from(from.parse()?)
        .to(to.parse()?)
        .subject(subject)
        .body(body.to_string())?;

    mailer.send(email).await?;

    Ok(())
}
