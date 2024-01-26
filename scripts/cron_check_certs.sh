#!/bin/bash

# Set the threshold for certificate expiration (in days)
THRESHOLD=30

# List of domains for which to check and renew certificates
DOMAINS=("kafka-cluster-1.staging.airt.ai")

sudo iptables -D INPUT -p tcp --dport 80 -j DROP

for domain in "${DOMAINS[@]}"; do
    # Check if certificate is expiring within the threshold
    expiration_date=$(date -d "$(sudo openssl x509 -in /etc/letsencrypt/live/$domain/fullchain.pem -noout -enddate | cut -d= -f2)" +%s)
    current_date=$(date +%s)
    days_until_expiry=$(( (expiration_date - current_date) / 86400 ))

    if [ "$days_until_expiry" -lt "$THRESHOLD" ]; then
        echo "Certificate for $domain is expiring in $days_until_expiry days. Renewing..."
        # sudo certbot renew --force-renewal
        sudo certbot renew --cert-name "$domain"
        if [ $? -eq 0 ]; then
            echo "Certificate renewed successfully for $domain."
            sudo cp /etc/letsencrypt/live/$domain/fullchain.pem /var/lib/redpanda/certs/broker.crt
            sudo cp /etc/letsencrypt/live/$domain/privkey.pem /var/lib/redpanda/certs/broker.key
            echo "Copied renewed certificates."
            sudo service redpanda restart
            echo "Redpanda service restarted."
        else
            echo "Certificate renewal failed for $domain."
        fi
    else
        echo "Certificate for $domain is not expiring within the next $THRESHOLD days. No action needed."
    fi
done

sudo iptables -A INPUT -p tcp --dport 80 -j DROP
