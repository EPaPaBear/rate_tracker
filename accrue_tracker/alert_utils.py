import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import tempfile
import os

def send_email_alert(subject, body, recipient, sender, smtp_server, smtp_port, smtp_user, smtp_pass, forecast_df=None):
    """
    Send email alert with optional CSV attachment.

    Args:
        subject: Email subject line
        body: Email body text
        recipient: Recipient email address
        sender: Sender email address
        smtp_server: SMTP server hostname
        smtp_port: SMTP server port
        smtp_user: SMTP username
        smtp_pass: SMTP password
        forecast_df: Optional DataFrame to attach as CSV

    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = subject

        # Attach body
        msg.attach(MIMEText(body, "plain"))

        # Attach forecast CSV if provided
        if forecast_df is not None and not forecast_df.empty:
            with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False, newline='') as tmp:
                forecast_df.to_csv(tmp.name, index=False)

            # Read the file back for attachment
            with open(tmp.name, 'rb') as attachment_file:
                attachment = MIMEBase("application", "octet-stream")
                attachment.set_payload(attachment_file.read())

            encoders.encode_base64(attachment)
            attachment.add_header(
                "Content-Disposition",
                "attachment; filename=forecast.csv"
            )
            msg.attach(attachment)

            # Clean up temp file
            os.unlink(tmp.name)

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(sender, recipient, msg.as_string())

        print(f"[INFO] Email sent successfully to {recipient}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        return False

def send_model_performance_alert(model_name, metrics, threshold, recipient, sender, smtp_config):
    """Send alert when model performance degrades below threshold."""
    subject = f"[ALERT] Model Performance Degraded - {model_name.upper()}"
    body = f"""Model Performance Alert

The {model_name.upper()} model is performing poorly and may need attention.

Current Metrics:
- MAE: {metrics['mae']:.4f}
- RMSE: {metrics['rmse']:.4f}
- MAPE: {metrics['mape']:.2f}%

Alert Threshold: {threshold}%

Please check the dashboard for more details and consider retraining the model.

This is an automated alert from the Cashramp Market Rate Tracker.
"""

    return send_email_alert(
        subject=subject,
        body=body,
        recipient=recipient,
        sender=sender,
        **smtp_config
    )

def send_rate_threshold_alert(alert_mode, threshold, min_rate, max_rate, horizon,
                            recipient, sender, smtp_config, forecast_df=None):
    """Send alert when forecasted rates breach user-defined thresholds."""
    direction = "below" if alert_mode.startswith("Below") else "above"
    extreme_rate = min_rate if direction == "below" else max_rate

    subject = f"[ALERT] Market Rate Forecast - Rate Going {direction.upper()} Threshold"
    body = f"""Market Rate Threshold Alert

The forecast predicts rates will go {direction} your defined threshold in the next {horizon} hours.

Alert Configuration:
- Mode: {alert_mode}
- Threshold: {threshold}
- Forecast Period: {horizon} hours

Forecast Summary:
- Minimum Rate: {min_rate:.4f}
- Maximum Rate: {max_rate:.4f}
- Extreme Rate: {extreme_rate:.4f}

{"" if forecast_df is None else "A detailed forecast CSV is attached for your analysis."}

Consider adjusting your trading strategy based on this forecast.

This is an automated alert from the Cashramp Market Rate Tracker.
"""

    return send_email_alert(
        subject=subject,
        body=body,
        recipient=recipient,
        sender=sender,
        forecast_df=forecast_df,
        **smtp_config
    )