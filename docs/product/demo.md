# CalHacks demo plan (draft for review, 2026-10-12)

Pitch line: "Anything a hearing person can do with their voice on a Mac, you can do by signing."
Say "any voice-driven app", never "any app".

## Setup on the table

- MacBook with the SignBridge menu-bar icon, preview window open on the left half of the screen:
  camera feed, tracking dots, rest line, the signs recognized so far, Claude's sentence, and the route.
- Target apps arranged on the right half: Siri, Messages (or Notes), and a voice call (FaceTime audio or
  Discord) with a second device so judges can hear the voice output.
- Good, even lighting; plain background; sit at the distance the onboarding calibrated.

## Script (about 3 minutes)

1. **Hook (20 s).** "Siri, Alexa, voice chat: the Mac assumes you can speak. Hundreds of thousands of people
   in the US use ASL. SignBridge lets them sign instead." (Verify a sourced figure before the demo.) Point at the preview: "This is what it sees."
2. **Siri (40 s).** Wake gesture -> sign RAIN TOMORROW -> preview shows "Will it rain tomorrow?  -> Siri".
   Thumbs up -> Siri opens and answers.
3. **Text (30 s).** Wake -> HUNGRY PIZZA -> "I'm hungry, want pizza?  -> Messages". Thumbs up -> the
   message appears in the chat.
4. **Safety (20 s).** Wake -> sign something, then cancel. Nothing happens. "It never acts without a
   confirm, and it never acts on a destructive command without one."
5. **Voice (40 s).** In the call: wake -> HELLO, then WAIT. The other device hears "Hello everyone!" and
   "Hold on a second." in a synthesized voice.
6. **How it works (30 s).** On-device tracking (MediaPipe), a small sign classifier trained on Deaf
   signers' data (PopSign) plus a few minutes of the user's own signing, Claude to turn glosses into
   sentences and pick the route. Landmarks only: no video leaves the Mac.
7. **Invite (rest).** Let a judge try HELLO or THANKYOU (PopSign-trained, so it works on new signers).

## Backup plan

- Record a backup video of the full script the day before, on the demo machine and lighting.
- If live recognition misbehaves: switch to the keyboard fake-sign mode (number keys inject sign
  sequences) so the Claude + routes half still runs live, and say so honestly.
- If Wi-Fi is bad: Claude needs the network. Keep a phone hotspot ready; cache intent results for the
  scripted phrases as a last resort and say so.

## Open questions for the user

- Which voice app for the call (FaceTime audio, Discord, Zoom)? Needs BlackHole selected as its mic.
- Is a second device available to play the call audio for judges?
- Messages vs Notes for the text scene (Messages needs a test contact).
