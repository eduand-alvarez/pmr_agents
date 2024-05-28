document.addEventListener('DOMContentLoaded', () => {
    const angleCtx = document.getElementById('angleChart').getContext('2d');
    const velocityCtx = document.getElementById('velocityChart').getContext('2d');
    const torqueCtx = document.getElementById('torqueChart').getContext('2d');
    const chatBox = document.getElementById('chat-box');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const armSelect = document.getElementById('robotic-arm-select');
    const maintenanceSelect = document.getElementById('maintenance-select');
    const notificationDropdown = document.getElementById('notification-dropdown');
    const notificationBtn = document.getElementById('notification-btn');

    const maintenanceFlaggedCounter = document.getElementById('maintenance-flagged');
    const maintenanceRequestsCounter = document.getElementById('maintenance-requests');
    const totalCostCounter = document.getElementById('total-cost');
    const progressBar = document.getElementById('progress-bar');
    const progressBarContainer = document.getElementById('progress-bar-container');

    let maintenanceFlaggedCount = 0;
    let maintenanceRequestsCount = 0;
    let totalCost = 0; // Placeholder value for total cost

    let currentData = null;
    let currentArm = armSelect.value;
    let ws = null;

    const popupContainer = document.getElementById('popup-container');
    const popupDetails = document.getElementById('popup-details');
    const popupClose = document.getElementById('popup-close');

    popupClose.addEventListener('click', () => {
        popupContainer.style.display = 'none';
    });

    window.showDetails = function(detailsId) {
        const details = window[detailsId];
        popupDetails.textContent = details;
        popupContainer.style.display = 'flex';
    };

    notificationBtn.addEventListener('click', () => {
        notificationDropdown.style.display = notificationDropdown.style.display === 'none' || notificationDropdown.style.display === '' ? 'block' : 'none';
    });

    const createChart = (ctx, label, borderColor, backgroundColor, yAxisLabel, chartTitle) => {
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: label,
                    data: [],
                    borderColor: borderColor,
                    backgroundColor: backgroundColor,
                    borderWidth: 2,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                elements: {
                    line: {
                        tension: 0.3 // smooth curves
                    },
                    point: {
                        radius: 0
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'second',
                            tooltipFormat: 'MMM DD, HH:mm:ss',
                            displayFormats: {
                                second: 'HH:mm:ss'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Time'
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: yAxisLabel
                        },
                        grid: {
                            color: '#eee'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: chartTitle
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.raw.toFixed(2);
                            }
                        }
                    }
                }
            }
        });
    };

    const angleChart = createChart(angleCtx, 'Joint Angle (degrees)', 'rgba(75, 192, 192, 1)', 'rgba(75, 192, 192, 0.2)', 'Angle (degrees)', 'Angle');
    const velocityChart = createChart(velocityCtx, 'Joint Velocity (deg/s)', 'rgba(153, 102, 255, 1)', 'rgba(153, 102, 255, 0.2)', 'Velocity (deg/s)', 'Velocity');
    const torqueChart = createChart(torqueCtx, 'Joint Torque (Nm)', 'rgba(255, 159, 64, 1)', 'rgba(255, 159, 64, 0.2)', 'Torque (Nm)', 'Torque');

    function connectWebSocket(armId) {
        if (ws) {
            ws.close();
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws?arm_id=${armId}`;
        ws = new WebSocket(wsUrl);

        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            if (message.type === 'data') {
                const data = message.data;
                const timestamp = new Date(data.timestamp * 1000);

                updateChart(angleChart, timestamp, data.joint_angle);
                updateChart(velocityChart, timestamp, data.joint_velocity);
                updateChart(torqueChart, timestamp, data.joint_torque);
            } else if (message.type === 'chat') {
                const chatMessage = document.createElement('div');
                chatMessage.className = 'chat-message system-message';
                chatMessage.innerHTML = message.message; // Use innerHTML to include the button
                chatBox.appendChild(chatMessage);
                chatBox.scrollTop = chatBox.scrollHeight;

                // Attach event listener to the schedule button
                const scheduleBtn = chatMessage.querySelector('.schedule-btn');
                if (scheduleBtn) {
                    scheduleBtn.addEventListener('click', () => {
                        const cost = parseFloat(scheduleBtn.getAttribute('data-cost'));
                        fetch('/schedule', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ cost: cost })
                        })
                        .then(response => response.json())
                        .then(data => {
                            const systemMessageElement = document.createElement('div');
                            systemMessageElement.className = 'chat-message system-message';
                            systemMessageElement.textContent = data.reply;
                            chatBox.appendChild(systemMessageElement);
                            chatBox.scrollTop = chatBox.scrollHeight;  // Scroll to the bottom

                            // Increment maintenance requests count
                            maintenanceRequestsCount++;
                            maintenanceRequestsCounter.textContent = maintenanceRequestsCount;

                            // Update total cost
                            totalCost = data.total_cost; // Update with actual cost
                            totalCostCounter.textContent = `$${totalCost.toFixed(2)}`;

                            // Show animation
                            showScheduleAnimation();

                        })
                        .catch(error => console.error('Error:', error));
                    });
                }

                // Attach event listener to the details button
                const detailsBtn = chatMessage.querySelector('.details-btn');
                if (detailsBtn) {
                    const detailsId = message.details_id;
                    window[detailsId] = message.retrieval_details;
                    detailsBtn.setAttribute('data-details', detailsId);
                }

                // Add notification
                addNotification(message.message);

                // Increment maintenance flagged count
                maintenanceFlaggedCount++;
                maintenanceFlaggedCounter.textContent = maintenanceFlaggedCount;
            }
        };
    }

    function updateChart(chart, timestamp, value) {
        chart.data.labels.push(timestamp);
        chart.data.datasets[0].data.push(value);

        if (chart.data.labels.length > 100) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }

        chart.update();
    }

    function showScheduleAnimation() {
        const animation = document.createElement('div');
        animation.className = 'schedule-animation';
        animation.innerText = 'Scheduled!';
        document.body.appendChild(animation);

        setTimeout(() => {
            animation.classList.add('fade-out');
            animation.addEventListener('animationend', () => {
                animation.remove();
            });
        }, 1000);
    }

    function addNotification(message) {
        const notificationItem = document.createElement('div');
        notificationItem.className = 'notification-item';
        notificationItem.textContent = `${new Date().toLocaleString()}: ${message}`;
        notificationDropdown.appendChild(notificationItem);
    }

    sendButton.addEventListener('click', () => {
        const userMessage = chatInput.value;
        if (userMessage.trim()) {
            const userMessageElement = document.createElement('div');
            userMessageElement.className = 'chat-message user-message';
            userMessageElement.textContent = `User: ${userMessage}`;
            chatBox.appendChild(userMessageElement);
            chatInput.value = '';
            chatBox.scrollTop = chatBox.scrollHeight;  // Scroll to the bottom

            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: userMessage })
            })
            .then(response => response.json())
            .then(data => {
                const systemMessageElement = document.createElement('div');
                systemMessageElement.className = 'chat-message system-message';
                systemMessageElement.textContent = `System: ${data.reply}`;
                chatBox.appendChild(systemMessageElement);
                chatBox.scrollTop = chatBox.scrollHeight;  // Scroll to the bottom
            })
            .catch(error => console.error('Error:', error));
        }
    });

    chatInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendButton.click();
        }
    });

    armSelect.addEventListener('change', (event) => {
        const selectedArmId = event.target.value;
        currentArm = selectedArmId;
        connectWebSocket(selectedArmId);
        resetCharts();
    });

    function resetCharts() {
        angleChart.data.labels = [];
        angleChart.data.datasets[0].data = [];
        velocityChart.data.labels = [];
        velocityChart.data.datasets[0].data = [];
        torqueChart.data.labels = [];
        torqueChart.data.datasets[0].data = [];
        angleChart.update();
        velocityChart.update();
        torqueChart.update();
    }

    // Initial connection
    connectWebSocket(currentArm);
});
