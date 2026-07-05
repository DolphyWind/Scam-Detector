# Scam-Detector
A Discord bot for detecting Mr. Beast scams and alike.

## How It Works
The bot uses a [DINOv3](https://ai.meta.com/research/dinov3/) model to extract features from
image attachments and compares them to known positives. Upon a match, a set of pre-defined
actions could be taken.

## Configuration
You can configure the bot with `/actions [add|list|remove|clear]` command. In `/action add`, some
actions may require an additional parameter `param`.

## Bot Link
You can invite the bot with [this](https://discord.com/oauth2/authorize?client_id=1523069858760097924) link.

## Privacy Notice
Images and messages are processesed for the sole purpose of scam detection and no message data is saved.
You can always self-host the bot freely.
