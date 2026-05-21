const slangInput = document.getElementById('slangInput');
const translatedOutput = document.getElementById('translatedOutput');
const statusText = document.getElementById('statusText');
const translateButton = document.getElementById('translateButton');
const clearButton = document.getElementById('clearButton');

const defaultOutput = 'Your translation will appear here.';

function setLoadingState(isLoading) {
	translateButton.disabled = isLoading;
	clearButton.disabled = isLoading;
	slangInput.disabled = isLoading;
	translateButton.textContent = isLoading ? 'Processing...' : 'Translate';
	statusText.textContent = isLoading ? 'Processing text...' : 'Ready to translate.';
}

function updateOutput(message) {
	translatedOutput.textContent = message;
}

async function translateText() {
	const inputValue = slangInput.value.trim();

	if (!inputValue) {
		updateOutput('Enter text to see a translation here.');
		statusText.textContent = 'Waiting for input.';
		return;
	}

	setLoadingState(true);
	updateOutput('Translating text...');

	try {
		const response = await fetch('/api/translate', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({ text: inputValue })
		});

		if (!response.ok) {
			throw new Error(`Request failed with status ${response.status}`);
		}

		const data = await response.json();
		const translatedText = data.translation || data.translatedText || data.output;

		if (!translatedText) {
			throw new Error('The backend did not return a translation.');
		}

		updateOutput(translatedText);
		statusText.textContent = 'Translation ready.';
	} catch (error) {
		updateOutput('Unable to translate right now. Please try again.');
		statusText.textContent = 'Translation failed.';
		console.error('Translation request failed:', error);
	} finally {
		setLoadingState(false);
	}
}

translateButton.addEventListener('click', translateText);

clearButton.addEventListener('click', () => {
	slangInput.value = '';
	statusText.textContent = 'Ready to translate.';
	updateOutput(defaultOutput);
	slangInput.focus();
});

document.querySelector('.back-button')?.addEventListener('click', () => {
	history.back();
});

setLoadingState(false);
updateOutput(defaultOutput);
