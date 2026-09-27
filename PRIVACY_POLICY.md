# Privacy Policy

*Last updated: 27.09.2026*

In this document, the usage of the terms "we" and "the Bot" is the same as in [TERMS_OF_SERVICE.md](TERMS_OF_SERVICE.md).

## What We Process
We only process the images sent to our servers **locally** to extract high-dimensional vector representations in order to perform a similarity search. We don't share your images with
third-parties for processing. Neither the images nor the vector representations are stored on our server. The known-scam database is curated by us. Images from your server are not added to it.

## What We Store
We only store the list of actions configured to be taken when a scam image is detected for your server. Additionally we log the message id (without guild or channel id) of the message
containing image and its similarity score. This data is not enough to identify you and only used to monitor the Bot's performance and to debug it.

## Why
This data is used only to provide the Bot's features or to improve them. We do not sell or share it with third-parties.

## Retention and deletion
You can delete your action data by removing all Bot actions associated with your server. Actions data is also deleted upon removing the Bot from your server. Logs are kept until the Docker
container for the Bot gets deleted when we redeploy or update the Bot. To request deletion of your data manually, contact us by privately messaging `dolphywind` on Discord. 

## Contact
Send a private message to `dolphywind` on Discord.
