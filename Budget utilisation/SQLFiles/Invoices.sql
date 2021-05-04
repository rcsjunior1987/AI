SELECT invoice.member_id
	 , invoice.invoice_number
     , invoice.invoice_date
     , invoice.invoice_total
     , invoice.state invoice_state
     , invoice.funded_total
     , invoice.is_reimbursement
     , invoice.created_at
     , invoice.submitted_date
     , invoice.approved_date
     , invoice.received_date
     , invoice.updated_at invoice_updated_at
     , invoice.provider_account_id
  FROM HH_invoice invoice
  INNER JOIN (
			   SELECT member.id member_id 
                    , MAX(plan.start_date) start_date
                    , plan.end_date
				 FROM HH_member member
                INNER JOIN HH_plan plan
                       ON plan.member_key = member.member_key
                          
			    WHERE plan.status = 'COMPLETED'  
				 GROUP BY plan.member_key
			 ) plans ON plans.member_id = invoice.member_id 
                   AND (invoice.created_at >= plans.start_date AND invoice.created_at <= plans.end_date)
 
 UNION
              
SELECT invoice.member_id
	 , invoice.invoice_number
     , invoice.invoice_date
     , invoice.invoice_total
     , invoice.state invoice_state
     , invoice.funded_total
     , invoice.is_reimbursement
     , invoice.created_at
     , invoice.submitted_date
     , invoice.approved_date
     , invoice.received_date
     , invoice.updated_at invoice_updated_at
     , invoice.provider_account_id
  FROM HH_invoice invoice
  INNER JOIN (
			   SELECT member.id member_id 
                    , MAX(plan.start_date) start_date
                    , plan.end_date
				 FROM HH_member member
                INNER JOIN HH_plan plan
                       ON plan.member_key = member.member_key
                          
			    WHERE plan.status = 'PLAN_DELIVERY_ACTIVE'  
				 GROUP BY plan.member_key
			 ) plans ON plans.member_id = invoice.member_id 
                   AND (invoice.created_at >= plans.start_date AND invoice.created_at <= plans.end_date)