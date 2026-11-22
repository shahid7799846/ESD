
if st.button("Check Email"):
    if email.strip() == "":
        st.warning("Please enter an email.")
    else:
        cleaned_email = clean_text(email)
        email_features = tfidf.transform([cleaned_email])

        # Predict email category
        prediction = model.predict(email_features)[0]

        # Calculate probabilities
        probabilities = model.predict_proba(email_features)[0]
        classes = model.classes_

        # Calculate confidence
        confidence = (
            probabilities[list(classes).index(prediction)] * 100
        )

        # Display prediction
        if prediction == "spam":
            st.error("This email is SPAM!")
        else:
            st.success("This email is NORMAL (HAM).")

        # Display confidence
        st.subheader("Prediction Confidence")
        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2f}%"
        )

        # Display class probabilities
        st.write("Class Probabilities:")

        for class_name, probability in zip(classes, probabilities):
            st.write(
                f"{class_name.upper()}: {probability * 100:.2f}%"
            )
