hive -e "
set mapred.job.queue.name=root.kuaicheshiyebu-houshichangyewuxian.amdwdev;
SELECT x.*, y.channel_id, y.sum_distinct_channel, y.sum_channel
FROM 
	(
	SELECT *
	FROM am_dw.ads_zs_users_info
	WHERE dt = 20180710
	)x
LEFT JOIN
	(
	SELECT z.driver_id AS driver_id, s.channel_id AS channel_id,sum_channel, sum_distinct_channel
	FROM
		(
		SELECT driver_id, COUNT(DISTINCT channel_id) AS sum_distinct_channel, COUNT(channel_id) AS sum_channel
		FROM langbi_dm.ods_binlog_t_leads_d_whole
		WHERE CONCAT(year,month,day)= 20180710
		GROUP BY driver_id
		)z
	LEFT JOIN
		(
		SELECT DISTINCT driver_id, channel_id
		FROM langbi_dm.ods_binlog_t_leads_d_whole
		WHERE CONCAT(year,month,day)= 20180710
		)s
	ON z.driver_id = s.driver_id
	)y
ON
x.pid = y.driver_id;">zs_users_plus_channel_id_feature.csv