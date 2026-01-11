# Signal Bot Infrastructure Setup Guide

A complete guide to deploying signal-cli-rest-api on Ubuntu for bot development.

## Prerequisites

- Ubuntu 22.04+ server with root/sudo access
- Docker and Docker Compose installed
- A phone number for Signal registration (can be VoIP in some cases)
- Static IP or domain (optional, for reliability)

## 1. Server Preparation

### Install Docker

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Verify installation
docker --version
docker compose version

# Log out and back in for group changes
```

### Create project directory

```bash
mkdir -p ~/signal-bot/{config,data,logs}
cd ~/signal-bot
```

## 2. Signal-CLI-REST-API Deployment

### Docker Compose configuration

Create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  signal-api:
    image: bbernhard/signal-cli-rest-api:latest
    container_name: signal-api
    restart: unless-stopped
    environment:
      - MODE=json-rpc
      - AUTO_RECEIVE_SCHEDULE=0 */5 * * * *
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./data/signal-cli:/home/.local/share/signal-cli
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

**Configuration notes**:
- `MODE=json-rpc` keeps a persistent JVM daemon for sub-second response times
- `AUTO_RECEIVE_SCHEDULE` syncs with Signal servers every 5 minutes (cron format)
- Port bound to localhost only—don't expose to internet
- Health check ensures automatic restart on failure

### Start the container

```bash
docker compose up -d
docker compose logs -f  # Watch startup logs
```

Wait for the message: `Started Application in X seconds`

## 3. Signal Account Registration

You have two options: register a new account (primary) or link to an existing account (secondary). **Linking is recommended** for bot development.

### Option A: Link as Secondary Device (Recommended)

This adds the bot to your existing Signal account. You can monitor bot conversations from your phone.

```bash
# Generate QR code link
curl -s "http://localhost:8080/v1/qrcodelink?device_name=stockbot" | qrencode -t ANSIUTF8

# Or open in browser:
# http://localhost:8080/v1/qrcodelink?device_name=stockbot
```

On your phone:
1. Open Signal → Settings → Linked Devices
2. Tap "Link New Device"
3. Scan the QR code

Verify linking:

```bash
curl http://localhost:8080/v1/accounts | jq
```

You should see your phone number in the response.

### Option B: Register New Account (Primary)

Use this for a dedicated bot number. Requires solving a CAPTCHA.

**Step 1: Get CAPTCHA token**

1. Open https://signalcaptchas.org/registration/generate.html in a browser
2. Complete the CAPTCHA
3. Open browser DevTools (F12) → Network tab
4. Look for the request to `signalcaptcha://` and copy the full token

**Step 2: Register**

```bash
PHONE="+15551234567"  # Your bot's phone number (E.164 format)
CAPTCHA="signalcaptcha://signal-recaptcha-v2.6L..."  # Full token

curl -X POST "http://localhost:8080/v1/register/${PHONE}" \
  -H "Content-Type: application/json" \
  -d "{\"captcha\": \"${CAPTCHA}\", \"use_voice\": false}"
```

**Step 3: Verify with SMS code**

```bash
curl -X POST "http://localhost:8080/v1/register/${PHONE}/verify/123456"
```

Replace `123456` with the code you received.

**Common registration issues**:
- "Invalid captcha": Token expired (use within 10 seconds) or IP mismatch
- 429 errors: Rate limited—wait 24 hours
- Voice verification: Set `use_voice: true` if SMS fails

## 4. Verify Installation

### Check account status

```bash
curl http://localhost:8080/v1/accounts | jq
```

### Send a test message

```bash
PHONE="+15551234567"  # Your bot's number
RECIPIENT="+15559876543"  # Your personal number

curl -X POST "http://localhost:8080/v2/send" \
  -H "Content-Type: application/json" \
  -d "{
    \"number\": \"${PHONE}\",
    \"recipients\": [\"${RECIPIENT}\"],
    \"message\": \"Hello from Signal Bot!\"
  }"
```

### Receive messages

```bash
curl "http://localhost:8080/v1/receive/${PHONE}" | jq
```

## 5. Webhook Configuration (For Real-time Messages)

Instead of polling, configure a webhook to receive messages instantly.

### Update docker-compose.yml

```yaml
services:
  signal-api:
    image: bbernhard/signal-cli-rest-api:latest
    container_name: signal-api
    restart: unless-stopped
    environment:
      - MODE=json-rpc
      - AUTO_RECEIVE_SCHEDULE=0 */5 * * * *
      - WEBHOOK_URL=http://stock-bot:5000/webhook
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./data/signal-cli:/home/.local/share/signal-cli
    networks:
      - bot-network

  stock-bot:
    build: ./bot
    container_name: stock-bot
    restart: unless-stopped
    depends_on:
      - signal-api
    environment:
      - SIGNAL_API_URL=http://signal-api:8080
      - SIGNAL_PHONE_NUMBER=+15551234567
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
    networks:
      - bot-network

networks:
  bot-network:
    driver: bridge
```

## 6. Security Hardening

### Firewall configuration

```bash
# Allow SSH
sudo ufw allow ssh

# Block Signal API from external access (it's localhost-only anyway)
sudo ufw enable
```

### Signal-CLI data backup

The `./data/signal-cli` directory contains your account keys. **Back this up securely.**

```bash
# Backup
tar -czvf signal-backup-$(date +%Y%m%d).tar.gz ./data/signal-cli

# Store encrypted
gpg -c signal-backup-*.tar.gz
```

### Log rotation

Create `/etc/logrotate.d/signal-bot`:

```
/home/youruser/signal-bot/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 youruser youruser
}
```

## 7. Systemd Service (Optional)

For non-Docker deployments or additional process management:

Create `/etc/systemd/system/signal-bot.service`:

```ini
[Unit]
Description=Signal Stock Bot
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/signal-bot
ExecStart=/usr/bin/docker compose up
ExecStop=/usr/bin/docker compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable signal-bot
sudo systemctl start signal-bot
```

## 8. Monitoring

### Basic health check script

Create `healthcheck.sh`:

```bash
#!/bin/bash

WEBHOOK_URL="https://your-alerting-service.com/webhook"  # Optional

check_signal_api() {
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/v1/health)
    if [ "$response" != "200" ]; then
        echo "$(date): Signal API unhealthy (HTTP $response)"
        # curl -X POST "$WEBHOOK_URL" -d '{"text": "Signal API down"}'
        docker compose restart signal-api
    fi
}

check_signal_api
```

Add to crontab:

```bash
*/5 * * * * /home/youruser/signal-bot/healthcheck.sh >> /home/youruser/signal-bot/logs/healthcheck.log 2>&1
```

## 9. Maintenance

### Update signal-cli-rest-api

Signal protocol changes require monthly updates:

```bash
cd ~/signal-bot
docker compose pull
docker compose up -d
docker compose logs -f signal-api  # Verify startup
```

### Check for protocol issues

If messages stop sending/receiving:

```bash
# Force re-sync with Signal servers
curl -X POST "http://localhost:8080/v1/receive/${PHONE}"

# Check for errors
docker compose logs signal-api | grep -i error
```

### Re-link if needed

If your linked device gets disconnected:

```bash
# Remove old registration
rm -rf ./data/signal-cli/*

# Restart and re-link
docker compose restart signal-api
# Then repeat the QR code linking process
```

## 10. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Account not found" | Registration incomplete | Re-register or re-link |
| Slow message delivery | `normal` mode instead of `json-rpc` | Set `MODE=json-rpc` |
| Missing group messages | GroupV2 sync needed | Call receive endpoint |
| "Untrusted identity" | Contact reinstalled Signal | Trust new identity key via API |
| Container OOM | JVM memory leak | Set `JAVA_OPTS=-Xmx512m` |
| CAPTCHA always fails | IP mismatch | Solve CAPTCHA from server IP |

## API Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Health check |
| `/v1/accounts` | GET | List registered accounts |
| `/v2/send` | POST | Send message |
| `/v1/receive/{number}` | GET | Receive pending messages |
| `/v1/groups/{number}` | GET | List groups |
| `/v1/qrcodelink` | GET | Generate linking QR |

Full API docs: http://localhost:8080/v1/api (Swagger UI when container running)

## Next Steps

With infrastructure running, proceed to the bot implementation in `DESIGN.md`.
